#/bin/bash
docker run --gpus all --shm-size 1g -p 8080:80 -v ~/.cache/huggingface/:/data ghcr.io/huggingface/text-generation-inference:3.0.1 --model-id meta-llama/Llama-3.2-3B-Instruct --num-shard 2 --enable-prefill-logprobs

