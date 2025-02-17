import argparse
import asyncio
import os
from random import Random
import re
from typing import Optional

from openai import AsyncOpenAI
from utils import write_jsonl, process_queue, llm_generate, UnifiedClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


supervisor_developer_prompt = """You are the supervisor of an apprentice assistant. You provide instructions and guidance to help them fulfill user requests efficiently and successfully.
No tools like calculator or web-search are available."""

supervisor_prompt_template = """<user_request>{0}</user_request>

Previous assistant thoughts (if any):
<thoughts>{1}</thoughts>

Instructions:
- Break down remaining work into small, atomic tasks
- Consider dependencies and optimal task ordering
- Provide clear, specific instructions for the next logical step

Based on the current state, provide the next instruction. Just generate the message for the assistant."""

check_completion_template = """<user_request>{0}</user_request>

Assistant's thoughts:
<thoughts>{1}</thoughts>

Classify the assistant's thoughts:
- If we are still in the middle of reasoning or the last verification failed, reply with <continue/>.
- If you suspect an error among the last thoughts, generate a verification instruction inside a <verify> tag.
- When the last thought contains a correct final answer, extract and copy it verbatim into an <output> tag.

Generate a single <continue/>, <verify> or <output> tag.
"""

assistant_developer_prompt = """You are a capable assistant focused on methodically processing user requests. Your goal is to execute the currrent instruction thoughtfully.
You don't have access to a calculator or web-search."""
assistant_prompt_template = """<user_request>{0}</user_request>

Progress notes:
<thoughts>{1}</thoughts>

Current instruction:
<instruction>{2}</instruction>

Focus on clear, actionable results, don't hallucinate. What is your single next immediate atomic thought?
Your answer text will automatically become a new thought item.
"""


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
    #return f'\n{"\n".join([f"<thought id=\"{i}\">{t}</thought>" for i, t in enumerate(thoughts)])}\n'
    return f'\n{"\n\n".join(thoughts)}\n'


def generate_supervisor_prompt(user_task: str, thoughts: list[str]) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    supervisor_prompt = supervisor_prompt_template.format(user_task, thoughts_formatted)
    # print("supervisor_prompt", supervisor_prompt)
    return generate_simple_request(supervisor_prompt, supervisor_developer_prompt)


def generate_assistant_prompt(
    user_task: str, step_instructions: str, thoughts: list[str]
) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    assistant_prompt = assistant_prompt_template.format(
        user_task, thoughts_formatted, step_instructions
    )
    # print("assistant_prompt", assistant_prompt)
    return generate_simple_request(assistant_prompt, assistant_developer_prompt)


def generate_check_completion_prompt(user_task: str, thoughts: list[str]) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    check_completion_prompt = check_completion_template.format(
        user_task, thoughts_formatted
    )
    # print("assistant_prompt", assistant_prompt)
    return generate_simple_request(check_completion_prompt, supervisor_developer_prompt)


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


async def generate_thought_sequence(
    task_prompt: str, 
    client: UnifiedClient,
    sampling_params: dict, 
    max_depth: int = 10
) -> None:
    instruction_trace = []
    thought_trace = []

    params_creative = add_temperature_setting(
        sampling_params, temperature=0.7, top_p=0.9
    )
    params_mild = add_temperature_setting(sampling_params, temperature=0.1, top_p=0.9)
    params_strict = add_temperature_setting(
        sampling_params, temperature=0.2, top_p=0.1
    )

    final_answer = None
    verification_instructions = None
    for i in range(max_depth):
        if verification_instructions is None:
            supervisor_prompt = generate_supervisor_prompt(
                task_prompt, thoughts=thought_trace
            )
            output = await llm_generate(client, supervisor_prompt, params_creative)
            next_step_instruction = output.choices[0].message.content.strip()
        else:
            next_step_instruction = verification_instructions

        next_step_instruction = next_step_instruction.strip()
        instruction_trace.append(next_step_instruction)
        print(f"Instruction: {next_step_instruction}\n\n")

        assistant_prompt = generate_assistant_prompt(
            task_prompt, next_step_instruction, thoughts=thought_trace
        )
        output = await llm_generate(client, assistant_prompt, params_mild)
        assistant_thought = output.choices[0].message.content.strip()
        thought_trace.append(assistant_thought)

        print(f"Thought: {assistant_thought}\n\n")
        check_completion_prompt = generate_check_completion_prompt(
            task_prompt, thoughts=thought_trace
        )
        output = await llm_generate(client, check_completion_prompt, params_strict)
        completion_status = output.choices[0].message.content

        match = re.search(
            r"<output>(.*?)</output>",
            completion_status,
            flags=re.DOTALL,
        )
        if match:
            final_answer = match.group(1)
            break
        match = re.search(
            r"<verify>(.*?)</verify>",
            completion_status,
            flags=re.DOTALL,
        )
        if match:
            verification_instructions = match.group(1)
        else:
            verification_instructions = None

    return final_answer, thought_trace, instruction_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="DeepInfra")
    parser.add_argument("--api_type", type=str, choices=["openrouter", "swissai", "local"], default="openrouter", help="API service to use")
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY", help="Environment variable name containing the API key")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-3.3-70b-instruct",
        help="Model to use. For SwissAI, use meta-llama/Llama-3.3-70B-Instruct format. For OpenRouter, use meta-llama/llama-3.3-70b-instruct format."
    )
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--output_jsonl", type=str, default="output.jsonl")
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    rng = Random(args.seed)

    # Configure base URL and API key based on API type
    if args.api_type == "swissai":
        base_url = "https://fmapi.swissai.cscs.ch"
        api_key_env = "SWISSAI_API_KEY"
        if "llama-3.3-70b-instruct" in args.model.lower():
            args.model = "meta-llama/Llama-3.3-70B-Instruct"  # SwissAI format
    else:
        base_url = args.base_url
        api_key_env = args.api_key_env

    # Create unified client
    client = UnifiedClient.create(
        api_type=args.api_type,
        model=args.model,
        base_url=base_url,
        api_key=os.getenv(api_key_env),
        timeout=args.timeout,
    )

    sampling_params = {
        "model": args.model,
        "max_tokens": 4096,
    }
    if args.api_type == "openrouter" and args.provider is not None:
        sampling_params["extra_body"] = {
            "provider": {
                "order": args.provider.split(","),
                "allow_fallbacks": False,
            },
        }

    # user_task_prompt = "43 + 1231 + 91 + 1 + 3 ="
    # user_task_prompt = "43 + 1231 + 91 + (1124 + 311) * -5 ="
    # gt = ["-5810"]

    # user_task_prompt = "Hello, how are you?"
    # user_task_prompt = "Write a nice poem about adding numbers."
    user_task_prompt = "92183 * 192281 ="
    gt = ["17725039423", "17,725,039,423"]

    async def sample_and_write_result(i: int):
        final_answer, thought_trace, instruction_trace = (
            await generate_thought_sequence(
                user_task_prompt, client, sampling_params, max_depth=8
            )
        )
        solved = False
        if final_answer:
            solved = any(x in final_answer for x in gt)

        data = {
            "solved": solved,
            "final_answer": final_answer,
            "user_task_prompt": user_task_prompt,
            "thought_trace": thought_trace,
            "instruction_trace": instruction_trace,
        }
        write_jsonl(args.output_jsonl, lines=[data], mode="a")
        print(f"{i}: solved={solved}, final_answer: {final_answer}")

    max_concurrent = args.max_concurrent
    await process_queue(
        ({"i": i} for i in range(100)),
        worker_func=sample_and_write_result,
        max_concurrent=max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
