import argparse
import asyncio
import os
from pathlib import Path
from random import Random
import re
import time
from typing import Any, Iterable, Iterator
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from utils import write_jsonl, process_queue


async def llm_generate(
    client: AsyncOpenAI,
    messages: Iterable[ChatCompletionMessageParam],
    sampling_params: dict[str, Any],
) -> ChatCompletion:
    max_retry = 3
    for trial in range(max_retry):
        try:
            return await client.chat.completions.create(
                extra_headers={"X-Title": "open-thought"},
                messages=messages,
                **sampling_params,
            )
        except Exception as e:
            print("failure response:", e)
            time.sleep(trial * trial)  # quadratic backoff
            if trial == max_retry - 1:
                raise


class BasicIntArithmeticTaskConfig:
    def __init__(
        self,
        min_digits: int = 1,
        max_digits: int = 5,
        min_terms: int = 2,
        max_terms: int = 8,
    ):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.operators = ["+", "-"]

    def validate(self):
        assert self.min_digits > 0
        assert self.max_digits >= self.min_digits
        assert self.min_terms > 1
        assert self.max_terms >= self.min_terms
        assert len(self.operators) > 0


def generate_task(rng: Random, cfg: BasicIntArithmeticTaskConfig) -> str:
    num_terms = rng.randint(cfg.min_terms, cfg.max_terms)
    num_digits = rng.randint(cfg.min_digits, cfg.max_digits)
    constants = [rng.randint(0, 10**num_digits) for _ in range(num_terms)]
    operators = [rng.choice(cfg.operators) for _ in range(num_terms - 1)]

    buffer = []

    ground_truth = constants[0]

    buffer.append(f"{constants[0]}")
    for i, op in enumerate(operators):
        c = constants[i + 1]
        buffer.append(op)
        buffer.append(f"{c}")

        if op == "+":
            ground_truth += c
        elif op == "-":
            ground_truth -= c
        else:
            RuntimeError("Unsupported operator")

    buffer.append(f"")

    question_templates = [
        "{0}",
        "{0} =",
        "{0} = ?",
        "What is {0}?",
        "Solve {0}",
        "Calculate {0}",
        # 'evaluate {0}',
        # 'do me a favor and calculate {0}',
        # 'Give me the result of {0}',
        # 'Help me solve this: {0}',
        # 'calculator: {0}',
        # 'Tell me the result of the following expression {0}',
    ]

    template = rng.choice(question_templates)
    task = " ".join(buffer)
    formatted_task = template.format(task)

    return formatted_task, str(ground_truth), num_terms, num_digits


def build_prompt(
    developer_prompt: str, task: str, developer_role: str = "system"
) -> list:
    messages = [
        {"role": developer_role, "content": developer_prompt},
        {
            "role": "user",
            "content": task,
        },
    ]
    return messages


class Stats:
    def __init__(self):
        self.started = 0
        self.completed = 0
        self.solved = 0


async def sample_concurrent(
    *,
    rng: Random,
    open_router_client: AsyncOpenAI,
    n: int,
    task_cfg: BasicIntArithmeticTaskConfig,
    developer_prompt: str,
    output_jsonl: str,
    sampling_params: dict,
    max_concurrent: int = 1,
):
    stats = Stats()

    def generate_sampling_jobs() -> Iterator[dict]:
        for i in range(n):
            task, y, num_terms, num_digits = generate_task(rng, task_cfg)
            x = build_prompt(developer_prompt=developer_prompt, task=task)
            yield (
                {
                    "index": i,
                    "input": x,
                    "target": y,
                    "num_terms": num_terms,
                    "num_digits": num_digits,
                }
            )

    async def sampling_worker(
        *, index: int, input: str, target: str, num_terms: int, num_digits: int
    ) -> str:

        output_match = False
        output_tag_found = False
        stats.started += 1
        try:
            output = await llm_generate(
                open_router_client,
                input,
                sampling_params,
            )

            response_text = output.choices[0].message.content
            finish_reason = output.choices[0].finish_reason
            provider = output.provider
          
            try:
                final_answer = re.search(
                    r"<final_answer>(.*?)</final_answer>",
                    response_text,
                    flags=re.DOTALL,
                ).group(1)
                output_tag_found = True

                if final_answer.find(target) >= 0:
                    output_match = True
            except:
                pass

            if output_match:
                stats.solved += 1

            log_data = {
                "solved": output_match,
                "output_tag_found": output_tag_found,
                "num_terms": num_terms,
                "num_digits": num_digits,
                "finish_reason": finish_reason,
                "ground_truth": target,
                "output": response_text,
                "input": input,
                "provider": provider,
                "sampling_params": sampling_params,
            }

            if output.usage is not None:
                log_data["completion_tokens"] = output.usage.completion_tokens
                log_data["prompt_tokens"] = output.usage.prompt_tokens

            write_jsonl(output_jsonl, [log_data], "a")

        except Exception as e:
            print("Sampling failed:", e)
            time.sleep(1.0)

        stats.completed += 1

        print(
            f"[{index}/{n}] output_tag_found={output_tag_found}, solved={output_match}, num_terms={num_terms}, num_digits={num_digits}, solve_rate={stats.solved}/{stats.completed} "
        )

    await process_queue(
        job_generator=generate_sampling_jobs(),
        worker_func=sampling_worker,
        max_concurrent=max_concurrent,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-3.3-70b-instruct"
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_jsonl", type=str, default="output.jsonl")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--system_prompt", type=str, default="./prompts/cot1.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=6)
    parser.add_argument("--min_terms", type=int, default=2)
    parser.add_argument("--max_terms", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    rng = Random(args.seed)

    open_router_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        timeout=args.timeout,
    )

    # meta-llama/llama-3.2-3b-instruct
    # meta-llama/llama-3.3-70b-instruct
    # qwen/qvq-72b-preview
    # google/gemini-2.0-flash-thinking-exp:free
    # deepseek/deepseek-chat

    sampling_params = {
        "model": args.model,
        "max_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        sampling_params["temperature"] = args.temperature
        sampling_params["top_p"] = args.top_p

    # https://openrouter.ai/docs/provider-routing#custom-routing
    if args.provider is not None:
        sampling_params["extra_body"] = {
            "provider": {
                "order": args.provider.split(","),
                "allow_fallbacks": False,
            },
        }

    prompt_filename = Path(
        args.system_prompt
    )  # e.g. source https://pastebin.com/dQG6JDV4
    developer_prompt = prompt_filename.read_text()

    task_cfg = BasicIntArithmeticTaskConfig(
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
    )

    await sample_concurrent(
        rng=rng,
        open_router_client=open_router_client,
        n=args.n,
        task_cfg=task_cfg,
        developer_prompt=developer_prompt,
        output_jsonl=args.output_jsonl,
        sampling_params=sampling_params,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
