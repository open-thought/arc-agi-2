#!/bin/bash
# run this script from the parent directory: scripts/sample_llama_3.2_3b_t5_64.sh
python sample_completions.py --temperature 0.5 --no_array_brackets --col_delimiter " " --row_delimiter $'\n' --trials 64 --jsonl-output-path output/llama-3.2-3B-instruct-stage0-simple-t5-64.jsonl --model-id meta-llama/Llama-3.2-3B-Instruct --num-shard 2 --eval-ids-json cfg/train_ids.json --max_concurrent 32
