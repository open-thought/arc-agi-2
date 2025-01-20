import argparse
import asyncio
import os
from random import Random
import re
from typing import Optional

from litellm import AsyncOpenAI
from utils import write_jsonl, process_queue, llm_generate


revision_agent_developer_prompt = """You are a stateful reasoning agent who iteratively progresses towards producing an excellent answer for given user requests. Your state is a thought trajectory which you can manipulate, e.g. by adding revising or deleting thoughts.
You don't have access to tools like web-search, shell or calculator. Use your state to reason and compute step-by-step."""

revision_next_step_prompt_template = """<user_request>{0}</user_request>

## Reasoning Trajectory
<thoughts>{1}</thoughts>

## Available Commands

- `<thought id="{{id}}">{{thought}}</thought>`  add a new thought with a new id or revise an existing thought be generating a thought tag with the same id
- `<delete id="{{id}}"/>`  delete an existing thought
- `<backtrack id="{{id}}/>`  remove all thoughts following the one with the specified id
- `<final_answer>{{output}}</final_answer>`  submit your answer when reasoning concluded

Note: {{thought}}, {{id}} and {{output}} are placeholders.

Break complex tasks recursively into manageable atomic thoughts, work logically step-by-step towards a solution. Generate a final answer when you feel confident about the result, otherwise revise or backtrack. Look for style hints in the user request when formulating the final answer.

Consider the user_request and the existing thoughts in the reasoning trajectory. Don't repeat existing thoughts, just generate a single next command tag."""


simple_agent_developer_prompt = """You are reasoning agent who iteratively progresses towards producing an excellent answer for given user requests. Your state is a thought trajectory.
You don't have access to tools like web-search, shell or calculator. Use your state to reason and compute step-by-step."""

simple_next_step_prompt_template = """<user_request>{0}</user_request>

## Reasoning Trajectory
<thoughts>{1}</thoughts>

## Available Commands

- `<thought>{{thought}}</thought>`  add a new thought
- `<final_answer>{{output}}</final_answer>`  submit your answer when reasoning concluded

Note: {{thought}} and {{output}} are placeholders.

Break complex tasks recursively into manageable atomic thoughts, work logically step-by-step towards a solution. Generate a final answer when you feel confident about the result, otherwise revise or backtrack. Look for style hints in the user request when formulating the final answer.

Consider the user_request and the existing thoughts in the reasoning trajectory. Don't repeat existing thoughts, just generate a single next command tag."""


def generate_simple_request(user_prompt: str, developer_prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": developer_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def format_thoughts(thoughts: list[str]) -> str:
    if not thoughts:
        return ""
    return f'\n{"\n".join([f"<thought id=\"{i}\">{t}</thought>" for i, t in enumerate(thoughts)])}\n'


def generate_step_prompt(
    user_task: str, thoughts: list[str], template: str, developer_prompt: str
) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    step_prompt = template.format(user_task, thoughts_formatted)
    return generate_simple_request(step_prompt, developer_prompt)


def add_temperature_setting(
    sampling_params: dict, temperature: float, top_p: Optional[float]
) -> dict:
    sampling_params = sampling_params.copy()
    sampling_params["temperature"] = temperature
    if temperature > 0 and top_p is not None:
        sampling_params["top_p"] = top_p
    elif "top_p" in sampling_params:
        del sampling_params["top_p"]
    return sampling_params


def parseInt(value) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


async def generate_thought_sequence_simple(
    task_prompt: str, client: AsyncOpenAI, sampling_params: dict, max_depth: int = 10
) -> None:
    thought_trace = []
    final_answer = None
    for i in range(max_depth):
        step_prompt = generate_step_prompt(
            task_prompt,
            thoughts=thought_trace,
            template=simple_next_step_prompt_template,
            developer_prompt=simple_agent_developer_prompt,
        )

        output = await llm_generate(client, step_prompt, sampling_params)
        output_text = output.choices[0].message.content

        thought_match = re.search(
            r"<thought(?:\sid=\"(.*?)\")?>(.*?)</thought>", output_text, flags=re.DOTALL
        )
        if thought_match:
            thought_content = thought_match.group(2)
            thought_trace.append(thought_content)

        final_answer_match = re.search(
            r"<final_answer>(.*?)</final_answer>",
            output_text,
            flags=re.DOTALL,
        )
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            # print(f"ANSWER: {final_answer}\n")
            break

    return final_answer, thought_trace


async def generate_thought_sequence_revision(
    task_prompt: str, client: AsyncOpenAI, sampling_params: dict, max_depth: int = 10
) -> None:
    thought_trace = []
    final_answer = None
    for i in range(max_depth * 2):
        if i % 2 == 0:
            step_prompt = generate_step_prompt(
                task_prompt,
                thoughts=thought_trace,
                template=revision_next_step_prompt_template,
                developer_prompt=revision_agent_developer_prompt,
            )

        else:
            step_prompt.append(
                {
                    "role": "assistant",
                    "content": output_text,
                }
            )
            step_prompt.append(
                {
                    "role": "user",
                    "content": """Do you see a thought that should be revised or deleted?
Check all thoughts for validity. If everything is fine respond with the word VALID.""",
                }
            )

        output = await llm_generate(client, step_prompt, sampling_params)
        output_text = output.choices[0].message.content

        thought_match = re.search(
            r"<thought(?:\sid=\"(.*?)\")?>(.*?)</thought>", output_text, flags=re.DOTALL
        )
        if thought_match:
            id = thought_match.group(1)
            pos = parseInt(id) if id else None

            thought_content = thought_match.group(2)
            if pos is None or pos < 0 or pos >= len(thought_trace):
                thought_trace.append(thought_content)
            else:
                # print(f"REVISION THOUGHT id={id}, content={thought_content}\n")
                thought_trace[pos] = thought_content

        delete_match = re.search(
            r"<delete id=\"(.*?)\"\s*/>",
            output_text,
            flags=re.DOTALL,
        )
        if delete_match:
            id = delete_match.group(1)
            # print(f"DELETE: id={id} (num_thoughts={len(thought_trace)})\n")
            pos = parseInt(id)
            if pos < len(thought_trace):
                del thought_trace[pos]

        backtrack_match = re.search(
            r"<backtrack id=\"(.*?)\"\s*/>",
            output_text,
            flags=re.DOTALL,
        )
        if backtrack_match:
            id = backtrack_match.group(1)
            print(f"BACKTRACK: id={id}\n")

        final_answer_match = re.search(
            r"<final_answer>(.*?)</final_answer>",
            output_text,
            flags=re.DOTALL,
        )
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            # print(f"ANSWER: {final_answer}\n")
            break

    return final_answer, thought_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="DeepInfra")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-3.3-70b-instruct"
    )
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument(
        "--output_jsonl", type=str, default="iterative_output_ll-3.3_70b.jsonl"
    )
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--num_tries", type=int, default=10)
    parser.add_argument(
        "--method", type=str, default="simple", help="either 'simple' or 'revision'"
    )
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

    sampling_params = {
        "model": args.model,
        "max_tokens": 4096,
    }
    if args.provider is not None:
        sampling_params["extra_body"] = {
            "provider": {
                "order": args.provider.split(","),
                "allow_fallbacks": False,
            },
        }

    # sampling_params = add_temperature_setting(
    #     sampling_params, temperature=0.8, top_p=0.9
    # )

    # user_task_prompt = "43 + 1231 + 91 + 1 + 3 ="
    # user_task_prompt = "43 + 1231 + 91 + (1124 + 311) * -5 ="
    # gt = ["-5810"]

    # user_task_prompt = "Hello, how are you?"
    # user_task_prompt = "Write a nice poem about adding numbers."
    user_task_prompt = "92183 * 192281 ="
    gt = ["17725039423", "17,725,039,423"]

    assert args.method in ("simple", "revision")
    print(f"Method: {args.method}")
    if args.method == "simple":
        thought_generator = generate_thought_sequence_simple

    elif args.method == "revision":
        thought_generator = generate_thought_sequence_revision
    else:
        raise RuntimeError("Unsupported method")

    async def sample_and_write_result(i: int):
        final_answer, thought_trace = await thought_generator(
            user_task_prompt,
            open_router_client,
            sampling_params,
            max_depth=args.max_depth,
        )
        solved = False
        if final_answer:
            solved = any(x in final_answer for x in gt)

        data = {
            "solved": solved,
            "final_answer": final_answer,
            "user_task_prompt": user_task_prompt,
            "thought_trace": thought_trace,
        }
        write_jsonl(args.output_jsonl, lines=[data], mode="a")
        print(
            f"{i}: solved={solved}, final_answer: {final_answer}, num_thoughts={len(thought_trace)}"
        )

    max_concurrent = args.max_concurrent
    await process_queue(
        ({"i": i} for i in range(args.num_tries)),
        worker_func=sample_and_write_result,
        max_concurrent=max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
