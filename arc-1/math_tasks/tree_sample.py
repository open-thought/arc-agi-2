import argparse
import asyncio
import os
from random import Random

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


assistant_developer_prompt = """You are a capable assistant focused on methodically processing user requests. Your goal is to execute the currrent instruction thoughtfully."""
assistant_prompt_template = """<user_request>{0}</user_request>

Progress notes:
<thoughts>{1}</thoughts>

Current instruction:
<instruction>{2}</instruction>

Focus on clear, actionable results. Your answer text will automatically become a new thought. Don't generate thought-tags, only the content.
"""


def generate_supervisor_prompt(user_task: str, thoughts: list[str]) -> list[dict]:
    thoughts_formatted = "\n".join(
        [f"<thought id={i}>\n{t}</thought>" for i, t in enumerate(thoughts)]
    )
    supervisor_prompt = supervisor_prompt_template.format(user_task, thoughts_formatted)
    # print("supervisor_prompt", supervisor_prompt)
    return [
        {
            "role": "system",
            "content": supervisor_developer_prompt,
        },
        {
            "role": "user",
            "content": supervisor_prompt,
        },
    ]


def generate_assistant_prompt(
    user_task: str, step_instructions: str, thoughts: list[str]
) -> list[dict]:
    thoughts_formatted = "\n".join(
        [f"<thought id={i}>\n{t}</thought>" for i, t in enumerate(thoughts)]
    )
    assistant_prompt = assistant_prompt_template.format(
        user_task, thoughts_formatted, step_instructions
    )
    # print("assistant_prompt", assistant_prompt)
    return [
        {
            "role": "system",
            "content": assistant_developer_prompt,
        },
        {
            "role": "user",
            "content": assistant_prompt,
        },
    ]


async def generate_thought_sequence(
    task_prompt: str, client: AsyncOpenAI, sampling_params: dict, max_depth: int = 10
) -> None:
    instruction_trace = []
    thought_trace = []

    sampling_params_creative = sampling_params.copy()
    sampling_params_creative["temperature"] = 0.5
    sampling_params_creative["top_p"] = 0.9

    for i in range(max_depth):
        supervisor_prompt = generate_supervisor_prompt(
            task_prompt, thoughts=thought_trace
        )
        output = await llm_generate(client, supervisor_prompt, sampling_params_creative)
        next_step_instruction = output.choices[0].message.content
        instruction_trace.append(next_step_instruction)

        # print(f"Instruction: {next_step_instruction}\n\n")

        assistant_prompt = generate_assistant_prompt(
            task_prompt, next_step_instruction, thoughts=thought_trace
        )
        output = await llm_generate(client, assistant_prompt, sampling_params)
        assistant_thought = output.choices[0].message.content
        thought_trace.append(assistant_thought)

        # print(f"Thought: {assistant_thought}\n\n")

    print(f"Task: {task_prompt}")
    for i in range(len(instruction_trace)):
        # print(f"Instruction {i}:", instruction_trace[i])
        print(f"[{i}]:", thought_trace[i])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-3.3-70b-instruct"
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

    #user_task_prompt = "43 + 1231 + 91 + 1 + 3 ="
    user_task_prompt = "43 + 1231 + 91 + (1124 + 311) * -5 ="
    # user_task_prompt = "Hello, how are you?"
    # user_task_prompt = "Write a nice poem about adding numbers."

    await generate_thought_sequence(
        user_task_prompt, open_router_client, sampling_params, max_depth=10
    )


if __name__ == "__main__":
    asyncio.run(main())
