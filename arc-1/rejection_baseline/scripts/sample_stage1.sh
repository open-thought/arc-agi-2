#!/bin/bash
# (run the parent directory)
python sample_completions.py --temperature 0.5 --no_array_brackets --col_delimiter " " --row_delimiter $'\n' --trials 16 --jsonl-output-path output/llama-3.2-3B-instruct-stage1-simple-t5-16.jsonl --model-id ../train/outputs/stage1/checkpoint-160/ --num-shard 2 --eval-ids-json cfg/train_ids.json --max_concurrent 32
