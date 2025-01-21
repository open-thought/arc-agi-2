from dataclasses import dataclass
import random
import re
from typing import Any, Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
)
from logging_utils import init_logger


logger = init_logger(__name__)


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=1.0,
        temperature=1.0,
        max_length=1024,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def get_sequence_log_probs(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    kl: Optional[torch.Tensor] = None


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience]) -> Experience:
    batch_data = {}
    keys = (
        "sequences",
        "action_log_probs",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def sample(self, batch_size: int) -> Experience:
        batch_items = random.sample(self.items, batch_size)
        return join_experience_batch(batch_items)

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]


def main():
    seed = 42
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    batch_size = 8

    device = torch.device("cuda", device_index)
    init_rng(seed)

    model, tokenizer = load_model(model_name, device_map=device)

    replay_buffer = ReplayBuffer(limit=128)

    # dummy experience
    sequence_ids, returns, action_mask, completions = rollout(
        model, tokenizer, "3 + 4 * 12 = ", "51", num_rollouts=8
    )

    eps = 1e-8
    advantages = (returns - returns.mean()) / (returns.std() + eps)

    print("completions", completions)
    print("returns", returns)
    print("advantage", advantages)
    print("action_mask", action_mask)

    attention_mask = sequence_ids != tokenizer.eos_token_id
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    logits = output["logits"]

    log_probs = get_sequence_log_probs(
        logits=logits[:, :-1].to(torch.float32), output_ids=sequence_ids[:, 1:]
    )

    experience = Experience(
        sequences=sequence_ids,
        action_log_probs=log_probs.detach(),
        returns=returns,
        advantages=advantages.detach(),
        attention_mask=attention_mask,
        action_mask=action_mask,
    )
    replay_buffer.append(experience)

    dataloader = DataLoader(
        replay_buffer,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=join_experience_batch,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for i in range(3):
        model.train()

        for experience in dataloader:
            optimizer.zero_grad()

            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

            output = model.forward(
                input_ids=sequence_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            logits = output["logits"]
            log_probs = get_sequence_log_probs(
                logits=logits[:, :-1].to(torch.float32), output_ids=sequence_ids[:, 1:]
            )

            old_log_probs = experience.action_log_probs
            advantages = experience.advantages
            action_mask = experience.action_mask
            clip_eps: float = 0.2

            # actor loss
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
            loss = -torch.min(surr1, surr2)

            # todo: add kl loss

            loss = masked_mean(loss, action_mask, dim=-1).mean()
            print(i, "loss", loss)

            loss.backward()

            optimizer.step()

    print(log_probs)


if __name__ == "__main__":
    main()
