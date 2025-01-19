import argparse
import asyncio
from enum import Enum
import os
from random import Random
import re
from typing import Optional, Self

from litellm import AsyncOpenAI
from utils import write_jsonl, process_queue, llm_generate
from mcts import NodeBase, MctsParamsBase


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


async def rollout_thought_sequence(
    task_prompt: str,
    client: AsyncOpenAI,
    sampling_params: dict,
    instruction_trace: Optional[list[str]],
    thought_trace: Optional[list[str]],
    max_depth: int = 10,
) -> None:
    instruction_trace = [] if instruction_trace is None else instruction_trace.copy()
    thought_trace = [] if thought_trace is None else thought_trace.copy()

    assert (
        len(instruction_trace) >= len(thought_trace)
        and len(instruction_trace) - len(thought_trace) <= 1
    )

    sampling_params_creative = sampling_params.copy()
    sampling_params_creative["temperature"] = 0.7
    sampling_params_creative["top_p"] = 0.9

    sampling_params_mild = sampling_params.copy()
    sampling_params_mild["temperature"] = 0.1
    sampling_params_mild["top_p"] = 0.9

    start_depth = len(thought_trace)
    if start_depth >= max_depth:
        return None, thought_trace, instruction_trace

    final_answer = None
    next_instructions = None
    if len(instruction_trace) > len(thought_trace):
        next_instructions = instruction_trace[-1]

    for i in range(start_depth, max_depth):
        if next_instructions is None:
            supervisor_prompt = generate_supervisor_prompt(
                task_prompt, thoughts=thought_trace
            )
            output = await llm_generate(
                client, supervisor_prompt, sampling_params_creative
            )
            instruction = output.choices[0].message.content
        else:
            instruction = next_instructions

        instruction_trace.append(instruction)
        # print(f"Instruction: {instruction}\n\n")

        assistant_prompt = generate_assistant_prompt(
            task_prompt, instruction, thoughts=thought_trace
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
            next_instructions = match.group(1)
        else:
            next_instructions = None

    return final_answer, thought_trace, instruction_trace


class NodeType(Enum):
    TASK_PROMPT = 1
    INSTRUCTION = 2
    THOUGHT = 3


class Node(NodeBase):
    def __init__(
        self, node_type: NodeType, params: MctsParamsBase, parent: Self = None
    ):
        super().__init__(params, parent)
        self.node_type = node_type
        self.content = None

    def traces(self) -> tuple[list[str], list[str]]:
        ancestors = list(self.ancestors())
        ancestors.reverse()
        instructions = []
        thoughts = []
        task_prompt = ancestors[0].content
        for i in range(1, len(ancestors)):
            ancestor = ancestors[i]
            if ancestor.content is None:
                assert len(ancestor.children) == 0
                break
            if i % 2 == 0:
                thoughts.append(ancestor.content)
            else:
                instructions.append(ancestor.content)
        return task_prompt, instructions, thoughts


class MctsSearch(MctsParamsBase):
    def __init__(
        self,
        task_prompt: str,
        ground_truth: str,
        client: AsyncOpenAI,
        sampling_params: dict,
        max_depth: int = 10,
    ):
        super().__init__()
        self.task_prompt = task_prompt
        self.ground_truth = ground_truth
        self.client = client
        self.sampling_params = sampling_params
        self.max_depth = max_depth

        self.root = Node(NodeType.TASK_PROMPT, params=self)
        self.root.visits = 1
        self.root.content = task_prompt

    def create_child(self, parent: Node) -> Node:
        child_type = (
            NodeType.INSTRUCTION
            if parent.node_type != NodeType.INSTRUCTION
            else NodeType.THOUGHT
        )
        return Node(node_type=child_type, params=self, parent=parent)

    async def step(self) -> None:
        node = self.root.select()
        print(
            f"root {{ value: {self.root.value}, max_children: {self.root.max_children} }}; "
            f"selected: depth={node.depth}, type={node.node_type}, visits={node.visits}, "
            f"value={node.value}, uct={node.uct}, content={node.content}"
        )
        task_prompt, instructions, thoughts = node.traces()

        # monte carlo rollout from here
        final_answer, thought_trace, instruction_trace = await rollout_thought_sequence(
            task_prompt,
            self.client,
            self.sampling_params,
            instructions,
            thoughts,
            max_depth=self.max_depth,
        )

        if node.content is None:
            if node.node_type == NodeType.INSTRUCTION:
                assert len(instruction_trace) > len(instructions)
                node.content = instruction_trace[len(instructions)]
            else:
                assert len(thought_trace) > len(thoughts)
                assert node.node_type == NodeType.THOUGHT
                node.content = thought_trace[len(thoughts)]
                if len(thought_trace) == len(thoughts) + 1:
                    if final_answer is not None or len(thought_trace) >= self.max_depth:
                        node.terminal = True

        # backprop
        if final_answer is not None and self.ground_truth in final_answer:
            node.update(1)
        else:
            node.update(0)


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
    user_task_prompt = "43 + 1231 + 91 + (1124 + 311) * -5 ="
    # user_task_prompt = "Hello, how are you?"
    # user_task_prompt = "Write a nice poem about adding numbers."

    s = MctsSearch(
        task_prompt=user_task_prompt,
        ground_truth="-5810",
        client=open_router_client,
        sampling_params=sampling_params,
        max_depth=8,
    )
    for i in range(100):
        await s.step()

    # async def sample_and_write_result(i: int):
    #     final_answer, thought_trace, instruction_trace = (
    #         await generate_thought_sequence(
    #             user_task_prompt, open_router_client, sampling_params, max_depth=8
    #         )
    #     )
    #     solved = False
    #     if final_answer:
    #         solved = any(x in final_answer for x in gt)

    #     data = {
    #         "solved": solved,
    #         "final_answer": final_answer,
    #         "user_task_prompt": user_task_prompt,
    #         "thought_trace": thought_trace,
    #         "instruction_trace": instruction_trace,
    #     }
    #     print(data)
    #     # write_jsonl(args.output_jsonl, lines=[data], mode="a")
    #     # print(f'{i}: solved={solved}, final_answer: {final_answer}')

    # max_concurrent = args.max_concurrent
    # await process_queue(
    #     ({"i": i} for i in range(1)),
    #     worker_func=sample_and_write_result,
    #     max_concurrent=max_concurrent,
    # )


if __name__ == "__main__":
    asyncio.run(main())
