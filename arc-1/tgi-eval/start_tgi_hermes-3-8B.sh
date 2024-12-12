#/bin/bash
sudo docker run --gpus all --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:3.0.0 --model-id NousResearch/Hermes-3-Llama-3.1-8B --num-shard 2
