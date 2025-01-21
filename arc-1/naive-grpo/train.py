import random
import re
from typing import Any
import torch
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


def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

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
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=1.0,
        temperature=1.0,
        max_length=1024,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    # 3. determine rewards
    rewards = torch.zeros(num_rollouts, dtype=torch.float)
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

        rewards[i] = reward

    return generated_ids, rewards.to(generated_ids.device), completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def main():
    seed = 42
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    device = torch.device("cuda", device_index)
    init_rng(seed)

    model, tokenizer = load_model(model_name, device_map=device)

    generated_ids, rewards, completions = rollout(
        model, tokenizer, "3 + 4 * 12 = ", "51", num_rollouts=8
    )
    advantage = (rewards - rewards.mean()) / rewards.std()

    print("completions", completions)
    print("rewards", rewards)
    print("advantage", advantage)
    
    attention_mask = generated_ids != tokenizer.eos_token_id
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(input_ids=generated_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits = output["logits"]
    print(logits.shape)


if __name__ == "__main__":
    main()
