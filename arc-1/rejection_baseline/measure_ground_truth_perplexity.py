import argparse
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Awaitable
import aiohttp
import asyncio
import json
import docker
from transformers import AutoTokenizer


async def generate_tgi(
    session: aiohttp.ClientSession,
    prompt: str,
    sampling_params: dict[str, Any],
    generate_url: str,
) -> dict[str, Any]:
    data = {"inputs": prompt, "parameters": sampling_params}
    async with session.post(generate_url, json=data) as r:
        r.raise_for_status()
        return await r.json()


def rfind_token_index(tokens: list[str], sub: str) -> int:
    combined = "".join(tokens)
    found_index = combined.rfind(sub)
    l = 0
    i = 0
    while l < found_index:
        l += len(tokens[i])
        i += 1
    return i


def range_perplexity_per_token(
    begin_pos: int, end_pos: int, tokens: list[str], logprobs: list[float]
) -> float:
    assert len(tokens) == len(logprobs)
    s = 0
    n = 0
    for i in range(begin_pos, end_pos):
        if tokens[i].isspace() or tokens[i] in (",", "[", "]"):
            continue
        s += logprobs[i]
        n += 1
    return math.exp(-s / n)


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()

    async with aiohttp.ClientSession() as session:

        # launch docker container and wait ready

        docker_container_name = "measure-ground-truth-tgi"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        num_shards = 2

        client = docker.from_env()
        hf_data_path = Path("~/.cache/huggingface/").expanduser()

        try:
            container = client.containers.get(docker_container_name)
            port = container.ports["80/tcp"][0]["HostPort"]
        except docker.errors.NotFound:
            port = random.randint(8000, 9000)
            container = client.containers.run(
                "ghcr.io/huggingface/text-generation-inference:3.0.1",
                command=[
                    "--model-id",
                    model_id,
                    "--num-shard",
                    str(num_shards),
                    "--enable-prefill-logprobs",
                ],
                shm_size="1G",
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,  # -1 means 'all available GPUs'
                        capabilities=[["gpu"]],  # Requests GPU devices
                    )
                ],
                detach=True,
                name=docker_container_name,
                auto_remove=True,
                ports={"80/tcp": port},
                volumes={str(hf_data_path): {"bind": "/data", "mode": "rw"}},
            )

        # wait for tgi to become ready
        base_url = f"http://localhost:{port}"
        generate_url = f"{base_url}/generate"

        max_wait_ready = 60
        for _ in range(max_wait_ready):
            try:
                async with session.get(url=f"{base_url}/health") as response:
                    response.raise_for_status()
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("TGI server launch timed out.")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        input_path = Path(
            "~/data/tgi-eval/llama_3.1_8B_t5_simple_64.jsonl"
        ).expanduser()

        parameters = {
            "details": True,
            "decoder_input_details": True,
            "do_sample": False,
            "max_new_tokens": 1,
        }

        with input_path.open(mode="r", encoding="utf-8") as f:
            s = f.readline()
            print(s)
            data_sample = json.loads(s)
            input = data_sample["input"]
            # append model response
            output_text = data_sample["output"]
            ground_truth = data_sample["ground_truth"]
            
            oracle_text = re.sub(
                r"<output>\s*(.*?)\s*</output>",
                f"<output>{ground_truth}</output>>",
                output_text,
                flags=re.DOTALL,
            )
            
            input.append({"role": "assistant", "content": oracle_text})

            # format prompt for model
            chat_prompt = tokenizer.apply_chat_template(input, tokenize=False)

            generate_result = await generate_tgi(
                session=session,
                prompt=chat_prompt,
                sampling_params=parameters,
                generate_url=generate_url,
            )

            print()
            prefill_details = generate_result['details']['prefill']
            prefill_tokens = [x['text'] for x in prefill_details]
            prefill_logprobs = [x['logprob'] for x in prefill_details]

            prefill_tokens

            begin_pos = rfind_token_index(prefill_tokens, "<output>")
            end_pos = rfind_token_index(prefill_tokens, "</output>")

            skip_begin = 3  # skip open tag tokens
            ppl = range_perplexity_per_token(
                begin_pos + skip_begin, end_pos, prefill_tokens, prefill_logprobs
            )

            print(ppl)


if __name__ == "__main__":
    asyncio.run(main())
