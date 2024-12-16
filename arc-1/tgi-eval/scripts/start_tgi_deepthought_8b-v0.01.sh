#/bin/bash
sudo docker run --gpus all --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:3.0.0 --model-id ruliad/deepthought-8b-llama-v0.01-alpha --num-shard 2
