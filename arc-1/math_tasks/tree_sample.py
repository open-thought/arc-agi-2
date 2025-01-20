import argparse
import asyncio
from enum import StrEnum
import os
from pathlib import Path
from random import Random
import re
from typing import Optional, Self
from uuid import uuid4

from litellm import AsyncOpenAI
from utils import write_json, write_jsonl, process_queue, llm_generate
from mcts import NodeBase, MctsParamsBase


supervisor_developer_prompt = """You are the supervisor of an apprentice assistant. You provide instructions and guidance to help them fulfill user requests efficiently and successfully.
No tools like calculator or web-search are available."""

supervisor_prompt_template = """<user_request>{0}</user_request>

Previous assistant thoughts (if any):
<thoughts>{1}</thoughts>

Instructions:
- Break down remaining work into small, manageable tasks
- Consider dependencies and optimal task ordering
- Provide clear, short instructions for the next logical step

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
You don't have a calculator tool, compute manually step-by-step."""
assistant_prompt_template = """<user_request>{0}</user_request>

Progress notes:
<thoughts>{1}</thoughts>

Current instruction:
<instruction>{2}</instruction>

Focus on clear, actionable results.
Your answer text will automatically become a new thought item. Think the next logical step!
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
    return f'\n{"\n\n".join(thoughts)}\n'


def generate_supervisor_prompt(user_task: str, thoughts: list[str]) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    supervisor_prompt = supervisor_prompt_template.format(user_task, thoughts_formatted)
    return generate_simple_request(supervisor_prompt, supervisor_developer_prompt)


def generate_assistant_prompt(
    user_task: str, step_instructions: str, thoughts: list[str]
) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    assistant_prompt = assistant_prompt_template.format(
        user_task, thoughts_formatted, step_instructions
    )
    return generate_simple_request(assistant_prompt, assistant_developer_prompt)


def generate_check_completion_prompt(user_task: str, thoughts: list[str]) -> list[dict]:
    thoughts_formatted = format_thoughts(thoughts)
    check_completion_prompt = check_completion_template.format(
        user_task, thoughts_formatted
    )
    return generate_simple_request(check_completion_prompt, supervisor_developer_prompt)


def add_temperature_setting(
    sampling_params: dict,
    temperature: float,
    top_p: Optional[float],
) -> dict:
    sampling_params = sampling_params.copy()
    sampling_params["temperature"] = temperature
    if temperature > 0 and top_p is not None:
        sampling_params["top_p"] = top_p
    elif "top_p" in sampling_params:
        del sampling_params["top_p"]
    return sampling_params


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

    params_creative = add_temperature_setting(
        sampling_params, temperature=0.7, top_p=0.9
    )
    params_mild = add_temperature_setting(sampling_params, temperature=0.1, top_p=0.9)
    params_strict = add_temperature_setting(
        sampling_params, temperature=0.2, top_p=0.1
    )

    start_depth = len(thought_trace)
    if start_depth >= max_depth:
        return None, thought_trace, instruction_trace

    final_answer = None
    next_instructions = None
    if len(instruction_trace) > len(thought_trace):
        next_instructions = instruction_trace.pop()

    for i in range(start_depth, max_depth):
        if next_instructions is None:
            supervisor_prompt = generate_supervisor_prompt(
                task_prompt, thoughts=thought_trace
            )
            output = await llm_generate(client, supervisor_prompt, params_creative)
            instruction = output.choices[0].message.content
        else:
            instruction = next_instructions

        instruction = instruction.strip()
        instruction_trace.append(instruction)
        # print(f"Instruction: {instruction}\n\n")

        assistant_prompt = generate_assistant_prompt(
            task_prompt, instruction, thoughts=thought_trace
        )
        output = await llm_generate(client, assistant_prompt, params_mild)
        assistant_thought = output.choices[0].message.content.strip()
        thought_trace.append(assistant_thought)

        # print(f"Thought: {assistant_thought}\n\n")
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
            next_instructions = match.group(1)
        else:
            next_instructions = None

    return final_answer, thought_trace, instruction_trace


class NodeType(StrEnum):
    TASK_PROMPT = "task_prompt"
    INSTRUCTION = "instruction"
    THOUGHT = "thought"


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

    def dump(self, depth: Optional[int] = None) -> dict:
        if depth is None or depth > 0:
            children_list = [
                child.dump(None if depth is None else depth - 1)
                for child in self.children
            ]
        else:
            children_list = None

        node_dict = {
            "visits": self.visits,
            "value": self.value,
            "score": self.value / self.visits if self.visits > 0 else None,
            "terminal": self.terminal,
            "uct": self.uct,
            "num_children": len(self.children),
            "type": str(self.node_type),
            "content": self.content,
            "children": children_list,
        }

        return node_dict


class MctsSearch(MctsParamsBase):
    def __init__(
        self,
        task_prompt: str,
        ground_truth: str,
        client: AsyncOpenAI,
        sampling_params: dict,
        rollouts_filename: Optional[str] = None,
        max_depth: int = 10,
        exploration_weight: float = 1.0,
        step_discount: float = 1.0,
    ):
        super().__init__(exploration_weight=exploration_weight, step_discount=step_discount)

        self.tree_id = uuid4()
        self.task_prompt = task_prompt
        self.ground_truth = ground_truth
        self.client = client
        self.sampling_params = sampling_params
        self.rollouts_filename = rollouts_filename
        self.max_depth = max_depth

        self.root = Node(NodeType.TASK_PROMPT, params=self)
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
        if node.terminal:
            node.update(node.value/node.visits)
            return

        print(
            f"[{self.tree_id}] value: {self.root.value}, visits={self.root.visits}, root_children: {len(self.root.children)}"
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
            solved = True
        else:
            node.update(0)
            solved = False

        if self.rollouts_filename:
            rollout_data = {
                "solved": solved,
                "start_depth": node.depth,
                "tree_id": str(self.tree_id),
                "final_answer": final_answer,
                "task_prompt": self.task_prompt,
                "thought_trace": thought_trace,
                "instruction_trace": instruction_trace,
            }
            write_jsonl(self.rollouts_filename, lines=[rollout_data], mode="a")

    def dump_tree(self, depth: Optional[int] = None) -> dict:
        tree_dict = {
            "id": str(self.tree_id),
            "exploration_weight": self.exploration_weight,
            "step_discount": self.step_discount,
            "alpha": self.alpha,
            "k": self.k,
            "root": self.root.dump(depth),
        }
        return tree_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="DeepInfra")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-3.3-70b-instruct"
    )
    parser.add_argument("--exploration", type=float, default=1.0, help="exploration weight")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--tree_dump_interval", type=int, default=10)
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

    sampling_params["extra_body"] = {
        "provider": {
            "require_parameters": True,
            "order": args.provider.split(","),
            "allow_fallbacks": False,
        },
    }
    if args.quantization:
        sampling_params["extra_body"]["provider"]["quantizations"] = args.quantization

    num_rollouts = args.num_rollouts
    max_depth = args.max_depth
    tree_dump_interval = args.tree_dump_interval
    output_path = Path(args.output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    tasks = [
        # {
        #     "id": "task_1",
        #     "task_prompt": "43 + 1231 + 91 + (1124 + 311) * -75 =",
        #     "ground_truth": "-106260",
        # },
        {
            "id": "task_2",
            "task_prompt": "92183 * 192281 =",
            "ground_truth": "17725039423",
        },
        # {
        #     "id": "task_3",
        #     "task_prompt": "Evaluate -(82194 + (19191+ 391+ 12+ 71)) =",
        #     "ground_truth": "-101859",
        # },
        {
            "id": "task_4",
            "task_prompt": "Calculate -(972647*268)-108 =",
            "ground_truth": "-260669504",
        },
        {
            "id": "task_5",
            "task_prompt": "What is 8376194 + 192 - (1092841 * 2 + 891901) ??",
            "ground_truth": "5298803",
        },
    ]

    async def sample_and_write_result(id: str, task_prompt: str, ground_truth: str):
        s = MctsSearch(
            task_prompt=task_prompt,
            ground_truth=ground_truth,
            client=open_router_client,
            sampling_params=sampling_params,
            rollouts_filename=output_path / f"{id}_rollouts.jsonl",
            max_depth=max_depth,
            exploration_weight=args.exploration,
            step_discount=0.95,
        )

        tree_filename = output_path / f"{id}_tree.json"
        for i in range(num_rollouts):
            try:
                await s.step()
                if (i + 1) % tree_dump_interval == 0:
                    write_json(tree_filename, s.dump_tree())
            except Exception as e:
                print("Sampling failed:", e)
                await asyncio.sleep(1.0)

        write_json(tree_filename, s.dump_tree())

    max_concurrent = args.max_concurrent
    await process_queue(
        (x for x in tasks),
        worker_func=sample_and_write_result,
        max_concurrent=max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
