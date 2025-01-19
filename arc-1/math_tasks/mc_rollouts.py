import argparse
import asyncio
import os
from random import Random
import re

from litellm import AsyncOpenAI
from utils import write_jsonl, process_queue, llm_generate


supervisor_developer_prompt = "You are the supervisor of an apprentice assistant. You provide instructions and guidance to help them fulfill user requests efficiently and successfully."
supervisor_prompt_template = """<user_request>{0}</user_request>

Previous assistant thoughts (if any):
<thoughts>{1}</thoughts>

Instructions:
- Break down remaining work into small, manageable tasks
- Consider dependencies and optimal task ordering
- Provide clear, specific instructions for the next logical step

Based on the current state, provide the next instruction. Just generate the message for the assistant."""


assistant_developer_prompt = """You are a capable assistant focused on methodically processing user requests. Your goal is to execute the currrent instruction thoughtfully.
You don't have a calculator tool, compute manually step-by-step."""
assistant_prompt_template = """<user_request>{0}</user_request>

Progress notes:
<thoughts>{1}</thoughts>

Current instruction:
<instruction>{2}</instruction>

Focus on clear, actionable results, don't hallucinate.
Your answer text will automatically become a new thought. Don't generate thought-tags, only the content.
"""


check_completion_template = """
<user_request>{0}</user_request>

Assistant's thoughts:
<thoughts>{1}</thoughts>

Classify the assistant's thoughts based on these rules:
- if we are still in the middle of the thought process or the last verification found the answer to be incorrect, reply with <continue/>.
- if there is an answer in last thoughts but no cross-check verification thought, generate a verification instructions inside a <verify> tag.
- when the last thought contains a verified correct result, extract the answer and copy it verabtim inside a <output> tag.

Answer with a single <verify>, <output> or <continue/> tag.
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
    return f'\n{"\n".join([f"<thought id=\"{i}\">{t}</thought>" for i, t in enumerate(thoughts)])}\n'


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


async def generate_thought_sequence(
    task_prompt: str, client: AsyncOpenAI, sampling_params: dict, max_depth: int = 10
) -> None:
    instruction_trace = []
    thought_trace = []

    sampling_params_creative = sampling_params.copy()
    sampling_params_creative["temperature"] = 0.7
    sampling_params_creative["top_p"] = 0.9

    sampling_params_mild = sampling_params.copy()
    sampling_params_mild["temperature"] = 0.1
    sampling_params_mild["top_p"] = 0.9

    final_answer = None
    verification_instructions = None
    for i in range(max_depth):
        if verification_instructions is None:
            supervisor_prompt = generate_supervisor_prompt(
                task_prompt, thoughts=thought_trace
            )
            output = await llm_generate(
                client, supervisor_prompt, sampling_params_creative
            )
            next_step_instruction = output.choices[0].message.content
        else:
            next_step_instruction = verification_instructions

        instruction_trace.append(next_step_instruction)
        # print(f"Instruction: {next_step_instruction}\n\n")

        assistant_prompt = generate_assistant_prompt(
            task_prompt, next_step_instruction, thoughts=thought_trace
        )
        output = await llm_generate(client, assistant_prompt, sampling_params_mild)
        assistant_thought = output.choices[0].message.content
        thought_trace.append(assistant_thought)

        # print(f"Thought: {assistant_thought}\n\n")
        check_completion_prompt = generate_check_completion_prompt(
            task_prompt, thoughts=thought_trace
        )
        output = await llm_generate(client, check_completion_prompt, sampling_params)
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
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-3.3-70b-instruct"
    )
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--output_jsonl", type=str, default="output.jsonl")
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

    # user_task_prompt = "43 + 1231 + 91 + 1 + 3 ="
    # user_task_prompt = "43 + 1231 + 91 + (1124 + 311) * -5 ="
    # user_task_prompt = "Hello, how are you?"
    # user_task_prompt = "Write a nice poem about adding numbers."
    user_task_prompt = "92183 * 192281 ="
    gt = ["17725039423", "17,725,039,423"]

    async def sample_and_write_result(i: int):
        final_answer, thought_trace, instruction_trace = await generate_thought_sequence(
            user_task_prompt, open_router_client, sampling_params, max_depth=8
        )
        solved = False
        if final_answer:
            solved = any(x in final_answer for x in gt)

        data = {
            'solved': solved,
            'final_answer': final_answer, 
            'user_task_prompt': user_task_prompt,
            'thought_trace': thought_trace,
            'instruction_trace': instruction_trace,
        }
        write_jsonl(args.output_jsonl, lines=[data], mode="a")
        print(f'{i}: solved={solved}, final_answer: {final_answer}')

    max_concurrent = args.max_concurrent
    await process_queue(({'i': i} for i in range(100)),  worker_func=sample_and_write_result, max_concurrent=max_concurrent)


if __name__ == "__main__":
    asyncio.run(main())
