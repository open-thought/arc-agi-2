import argparse
from pathlib import Path
import re
from typing import Optional
import aiohttp
import asyncio
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils import (
    write_jsonl,
    read_jsonl,
    process_queue,
    rfind_token_index,
    range_perplexity_per_token,
)
import tgi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shard", type=int, default=1)
    parser.add_argument(
        "--model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument(
        "--input-path",
        type=str,
        default="~/data/tgi-eval/llama_3.1_8B_t5_simple_64.jsonl",
    )
    parser.add_argument(
        "--output-path", type=str, default="llama_3.1_8B_t5_simple_64_scores.jsonl"
    )
    args = parser.parse_args()
    return args


async def determine_oracle_perplexity(
    *,
    session: aiohttp.ClientSession,
    tokenizer: PreTrainedTokenizerBase,
    input: list[dict],
    output: str,
    ground_truth: str,
    generate_url: str,
) -> Optional[float]:
    # append model response with oracle output
    oracle_text = re.sub(
        r"<output>\s*(.*?)\s*</output>",
        f"<output>{ground_truth}</output>>",
        output,
        flags=re.DOTALL,
    )

    input = input.copy()
    input.append({"role": "assistant", "content": oracle_text})

    # format prompt for model
    chat_prompt = tokenizer.apply_chat_template(input, tokenize=False)

    sampling_params = {
        "details": True,
        "decoder_input_details": True,
        "max_new_tokens": 1,
    }

    generate_result = await tgi.generate(
        session=session,
        prompt=chat_prompt,
        sampling_params=sampling_params,
        generate_url=generate_url,
    )

    prefill_details = generate_result["details"]["prefill"]
    prefill_tokens = [x["text"] for x in prefill_details]
    prefill_logprobs = [x["logprob"] for x in prefill_details]

    begin_pos = rfind_token_index(prefill_tokens, "<output>")
    end_pos = rfind_token_index(prefill_tokens, "</output>")
    if end_pos < begin_pos or begin_pos < 0 or end_pos < 0:
        return None

    skip_begin = 3  # skip open tag tokens
    ppl = range_perplexity_per_token(
        begin_pos + skip_begin, end_pos, prefill_tokens, prefill_logprobs
    )

    return ppl


async def main():
    args = parse_args()
    model_id = args.model_id
    docker_container_name = "tgi_inference"
    input_path = Path(args.input_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    http_timeout = aiohttp.ClientTimeout(total=600, sock_connect=60)
    async with aiohttp.ClientSession(timeout=http_timeout) as session:

        # launch tgi docker container and wait until it becomes ready
        port = tgi.start_container(
            container_name=docker_container_name,
            model_id=model_id,
            num_shard=args.num_shard,
        )
        base_url = f"http://localhost:{port}"
        await tgi.until_ready(session, base_url, max_tries=90)

        generate_url = f"{base_url}/generate"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        async def sampling_worker(
            *,
            id: str,
            trial: int,
            input: dict,
            output: str,
            ground_truth: str,
            output_tag_found: bool,
            **kwargs,
        ) -> Optional[dict]:
            if not output_tag_found:
                return None

            ppl = await determine_oracle_perplexity(
                session=session,
                tokenizer=tokenizer,
                input=input,
                output=output,
                ground_truth=ground_truth,
                generate_url=generate_url,
            )
            if ppl is None:
                return None

            result = {"id": id, "trial": trial, "ppl": ppl}
            write_jsonl(output_path, [result])
            return result

        results = await process_queue(
            job_generator=read_jsonl(input_path),
            worker_func=sampling_worker,
            max_concurrent=args.max_concurrent,
        )


if __name__ == "__main__":
    asyncio.run(main())
