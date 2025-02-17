import asyncio
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator, Optional, Union
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
import torch.cuda

# Helper classes for response formatting
class Message:
    def __init__(self, content):
        self.content = content

class Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason

class Usage:
    def __init__(self, completion_tokens, prompt_tokens):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens

class LocalChatCompletion:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage

@dataclass
class UnifiedClient:
    """Wrapper class to provide unified interface for both API and local models"""
    client: Union[AsyncOpenAI, tuple[Any, Any]]
    api_type: str

    @classmethod
    def create(
        cls,
        api_type: str,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> "UnifiedClient":
        """Create a unified client for either local or API-based models"""
        if api_type == "local":
            try:
                print(f"Loading local model {model}", end="\r")

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model,
                    local_files_only=True,
                    add_bos_token=True,
                    add_eos_token=True,
                    padding_side='left'
                )

                # Set padding token to EOS token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Configure model for inference
                model_config = {
                    "device_map": "auto",
                    "local_files_only": True,
                    # "use_cache": True
                }

                # Load the model
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model,
                    **model_config
                ).eval()

                print(f"\033[KLoaded local model {model}")
                return cls(client=(tokenizer, model_obj), api_type=api_type)
            except Exception as e:
                print(f"utils.py: Error during model/tokenizer loading: {str(e)}")
                raise
        else:
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            return cls(client=client, api_type=api_type)

async def local_llm_generate(
    model,
    tokenizer,
    messages: list,  # (list of dict) or batch (list of list of dict)
    sampling_params: dict
) -> Any:
    """Generate text using a local model, supporting batched input when messages is a list of message lists"""
    try:
        def prepare_text(msgs: list[dict]) -> str:
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                text_str = ""
                for msg in msgs:
                    if msg["role"] == "system":
                        text_str += f"System: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        text_str += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        text_str += f"Assistant: {msg['content']}\n"
                text_str += "Assistant: "
                return text_str
            else:
                return tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )

        def generate_completion(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(): # torch.amp.autocast('cuda'):
                return model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=sampling_params.get("max_tokens", 4096),
                    temperature=sampling_params.get("temperature", 0.1),
                    top_p=sampling_params.get("top_p", 0.9),
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=sampling_params.get("temperature", 0.1) > 0,
                    # use_cache=True
                )

        def create_completion(output: torch.Tensor, input_length: int) -> LocalChatCompletion:
            response = tokenizer.decode(output[input_length:])
            generated_tokens = len(output) - input_length
            return LocalChatCompletion(
                choices=[Choice(
                    message=Message(content=response.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")),
                    finish_reason = "length" if generated_tokens >= sampling_params.get("max_tokens", 4096) else "stop" if tokenizer.eos_token in response else "unknown"
                )],
                usage=Usage(
                    completion_tokens=generated_tokens,
                    prompt_tokens=input_length
                )
            )

        is_batch = isinstance(messages, list) and messages and isinstance(messages[0], list)
        texts = [prepare_text(msg) for msg in messages] if is_batch else [prepare_text(messages)]

        # Tokenize inputs
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sampling_params.get("max_tokens", 4096)
        )

        attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        attention_mask = attention_mask.to(model.device)
        outputs = generate_completion(inputs["input_ids"], attention_mask)

        # Create completion objects
        completions = [
            create_completion(output, len(inputs["input_ids"][i]))
            for i, output in enumerate(outputs)
        ]

        return completions if is_batch else completions[0]

    except Exception as e:
        print(f"utils.py: Error during generation: {str(e)}")
        if hasattr(model, 'config'):
            print(f"utils.py: Model config: {model.config}")
        raise

async def llm_generate(
    client: UnifiedClient,
    messages: Iterable[ChatCompletionMessageParam],
    sampling_params: dict[str, Any]
) -> ChatCompletion:
    """Unified generation function that works with both API and local models"""
    if client.api_type == "local":
        tokenizer, model = client.client
        return await local_llm_generate(model, tokenizer, messages, sampling_params)

    # API client
    max_retry = 10
    for trial in range(max_retry):
        try:
            headers = {"X-Title": "open-thought"} if client.api_type == "openrouter" else None
            return await client.client.chat.completions.create(
                extra_headers=headers,
                messages=messages,
                **sampling_params,
            )
        except Exception as e:
            print("failure response:", e)
            await asyncio.sleep(min(60, 10*trial))
            if trial == max_retry - 1:
                raise

async def process_queue(
    job_generator: Iterator[Any],
    worker_func: Callable,
    max_concurrent: int = 3,
    batch_enabled: bool = False
) -> list:
    """
    Process jobs with limited concurrency, supporting both single and batched processing.
    
    Args:
        job_generator: Iterator yielding jobs to process
        worker_func: Async function to process each job or batch of jobs
        max_concurrent: Maximum number of concurrent jobs/batches
        batch_enabled: Whether to batch jobs (True for local client with max_concurrent > 1)
    """
    pending = set()
    results = []
    jobs = job_generator

    if batch_enabled:
        jobs = []
        current_batch = []

        for job in job_generator:
            current_batch.append(job)
            if len(current_batch) >= max_concurrent:
                jobs.append(current_batch)
                current_batch = []
        if current_batch:  # Don't forget the last partial batch
            jobs.append(current_batch)

    async def run_job(job):
        try:
            if batch_enabled:
                # For batched mode, job is a list of jobs
                result = await worker_func(**{"batch": job} if len(job) > 1 else job[0])
            else:
                # For non-batched mode, job is a single job dict
                result = await worker_func(**job)

            # Handle both single results and batch results
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        finally:
            pending.discard(task)

    try:
        for job in jobs:
            task = asyncio.create_task(run_job(job))
            pending.add(task)

            if len(pending) >= max_concurrent:
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )
                pending -= done

        if pending:
            await asyncio.wait(pending)

    except Exception as e:
        for task in pending:
            task.cancel()
        raise e

    return results

def write_jsonl(file_name: str | Path, lines: list, mode: str = "a") -> None:
    file_path = Path(file_name)
    with file_path.open(mode, encoding="utf-8") as f:
        for l in lines:
            json.dump(l, f)
            f.write("\n")

def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def write_json(file_name: str | Path, data: Any) -> None:
    file_path = Path(file_name)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

def read_json(file_name: str | Path) -> Any:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        return json.load(f)