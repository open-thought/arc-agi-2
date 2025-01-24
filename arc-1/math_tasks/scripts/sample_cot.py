import argparse
import asyncio
import os
from pathlib import Path
from random import Random
import re
import time
from typing import Any, Iterable, Iterator
from openai import AsyncOpenAI
from utils import write_jsonl, process_queue, llm_generate
from tqdm import tqdm


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
    client: AsyncOpenAI,
    n: int,
    task_cfg: BasicIntArithmeticTaskConfig,
    developer_prompt: str,
    output_jsonl: str,
    sampling_params: dict,
    max_concurrent: int = 1,
    api_type: str,
):
    stats = Stats()
    pbar = tqdm(total=n, desc="Sampling progress")

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
        start_time = time.time()
        try:
            output = await llm_generate(
                client,
                input,
                sampling_params,
                api_type,
            )

            response_text = output.choices[0].message.content
            finish_reason = output.choices[0].finish_reason
            provider = getattr(output, 'provider', api_type)
            completion_time = time.time() - start_time
          
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
                "completion_time": completion_time,
            }

            if output.usage is not None:
                log_data["completion_tokens"] = output.usage.completion_tokens
                log_data["prompt_tokens"] = output.usage.prompt_tokens

            write_jsonl(output_jsonl, [log_data], "a")

        except Exception as e:
            print("sample_cot: Sampling failed:", e)
            time.sleep(1.0)

        stats.completed += 1
        pbar.update(1)
        pbar.set_postfix({
            'solved': f"{stats.solved}/{stats.completed}",
            'rate': f"{stats.solved/stats.completed:.1%}" if stats.completed > 0 else "0%"
        })

        print(
            f"[{index}/{n}] output_tag_found={output_tag_found}, solved={output_match}, num_terms={num_terms}, num_digits={num_digits}, solve_rate={stats.solved}/{stats.completed} "
        )

    try:
        await process_queue(
            job_generator=generate_sampling_jobs(),
            worker_func=sampling_worker,
            max_concurrent=max_concurrent,
        )
    finally:
        pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_type", type=str, choices=["openrouter", "swissai"], default="openrouter", help="API service to use")
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY", help="Environment variable name containing the API key")
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/llama-3.3-70b-instruct",
        help="Model to use. For SwissAI, use meta-llama/Llama-3.3-70B-Instruct format. For OpenRouter, use meta-llama/llama-3.3-70b-instruct format."
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_jsonl", type=str, default="output.jsonl")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--system_prompt", type=str, default="cot1.txt")
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

    try:
        dir = Path(__file__).parent.parent
    except:
        dir = Path("..")
    prompt_filename = str(dir / "prompts" / args.system_prompt)
    Path(prompt_filename).parent.mkdir(parents=True, exist_ok=True)
    prompt_file_path = Path(prompt_filename)
    output_filename = str(dir / "data" / args.output_jsonl)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    output_file_path = Path(output_filename)

    if args.api_type == "swissai":
        base_url = "https://fmapi.swissai.cscs.ch"
        api_key_env = "SWISSAI_API_KEY"
        if "llama-3.3-70b-instruct" in args.model.lower():
            args.model = "meta-llama/Llama-3.3-70B-Instruct"  # SwissAI format
    else:
        base_url = args.base_url
        api_key_env = args.api_key_env

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv(api_key_env),
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

    # Add OpenRouter-specific provider routing
    if args.api_type == "openrouter" and args.provider is not None:
        sampling_params["extra_body"] = {
            "provider": {
                "order": args.provider.split(","),
                "allow_fallbacks": False,
            },
        }

    developer_prompt = prompt_file_path.read_text() # e.g. source https://pastebin.com/dQG6JDV4

    task_cfg = BasicIntArithmeticTaskConfig(
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
    )

    await sample_concurrent(
        rng=rng,
        client=client,
        n=args.n,
        task_cfg=task_cfg,
        developer_prompt=developer_prompt,
        output_jsonl=output_file_path,
        sampling_params=sampling_params,
        max_concurrent=args.max_concurrent,
        api_type=args.api_type,
    )


if __name__ == "__main__":
    asyncio.run(main())
