#!/bin/bash
# 1.) pass@1, argmax
python -u sample_completions.py --model-id ../train/outputs/stage2/checkpoint-62/ --no_array_brackets --col_delimiter " " --row_delimiter $'\n' --num-shard 2 --max_concurrent 16 --jsonl-output-path output/eval-stage2-pass1.jsonl --eval-ids cfg/eval_ids.json --trials 1 --temperature 0
# 2.) pass@3, temperature 1.0
python -u sample_completions.py --model-id ../train/outputs/stage2/checkpoint-62/ --no_array_brackets --col_delimiter " " --row_delimiter $'\n' --num-shard 2 --max_concurrent 16 --jsonl-output-path output/eval-stage2-pass3.jsonl --eval-ids cfg/eval_ids.json --trials 3 --temperature 1.0
