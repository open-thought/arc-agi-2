from pathlib import Path
import random
import time
from typing import Any
import aiohttp
import docker
from huggingface_hub.constants import HF_HOME


async def generate(
    session: aiohttp.ClientSession,
    prompt: str,
    sampling_params: dict[str, Any],
    generate_url: str,
) -> dict[str, Any]:
    data = {"inputs": prompt, "parameters": sampling_params}
    async with session.post(generate_url, json=data) as r:
        r.raise_for_status()
        return await r.json()


def start_container(
    container_name: str,
    model_id: str,
    num_shard: int,
    image_name: str = "ghcr.io/huggingface/text-generation-inference:3.0.1",
    auto_remove: bool = True,
) -> int:
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        port = container.ports["80/tcp"][0]["HostPort"]
    except docker.errors.NotFound:
        port = random.randint(8000, 9000)

        volumes = {HF_HOME: {"bind": "/data", "mode": "rw"}}

        if Path(model_id).exists():
            volumes[model_id] = {"bind": "/data/model", "mode": "rw"}
            model_id = "/data/model"

        container = client.containers.run(
            image=image_name,
            command=[
                "--model-id",
                model_id,
                "--num-shard",
                str(num_shard),
                "--enable-prefill-logprobs",
                "--max-concurrent-requests",
                "1024",
            ],
            shm_size="1G",
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,  # -1 means 'all available GPUs'
                    capabilities=[["gpu"]],  # Requests GPU devices
                )
            ],
            detach=True,
            name=container_name,
            auto_remove=auto_remove,
            ports={"80/tcp": port},
            volumes=volumes,
        )
    return port


async def until_ready(
    session: aiohttp.ClientSession, base_url: str, max_tries: int = 90
) -> None:
    for _ in range(max_tries):
        try:
            async with session.get(url=f"{base_url}/health") as response:
                response.raise_for_status()
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("TGI server launch timed out.")
