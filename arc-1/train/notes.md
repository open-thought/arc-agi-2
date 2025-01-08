
# Axolotl

- [Documentation](https://axolotl-ai-cloud.github.io/axolotl/)
  - [List of Axolotl config options](https://axolotl-ai-cloud.github.io/axolotl/docs/config.html)

### Install Axolotl

`pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]`

install examples: `axolotl fetch examples`
run llama vision example: `axolotl train examples/llama-3-vision/lora-11b.yaml`

### Launch llama-3-vision example training

llama-3-vision example configuration: [llama-3-vision/lora-11b.yaml](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/llama-3-vision/lora-11b.yaml)

Dataset used in example (with images): [HuggingFaceH4/llava-instruct-mix-vsft](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)

- vision prompt description in llama-models gh repo [vision_prompt_format.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/vision_prompt_format.md)
    - "We recommend using a single image in one prompt"

- chat template in axolotl [llama3_2_vision chat template](https://github.com/axolotl-ai-cloud/axolotl/blob/bd2a594b8954103719f8d1ef739e2c3267ca36f6/src/axolotl/utils/chat_templates.py#L26)

### Example dataset entry

A dataset format compatible to the TRL SFT Trainer is used.

The `messages` field is a (list).

```json
[
    { "content": [ { "index": null, "text": "Who wrote this book?\n", "type": "text" }, { "index": 0, "text": null, "type": "image" } ], "role": "user" },
    { "content": [ { "index": null, "text": "Donna Eden", "type": "text" } ], "role": "assistant" },
    { "content": [ { "index": null, "text": "What is the title of this book?", "type": "text" } ], "role": "user" },
    { "content": [ { "index": null, "text": "The Energies of Love: Using Energy Medicine to Keep Your Relationship Thriving", "type": "text" } ], "role": "assistant" },
    { "content": [ { "index": null, "text": "What type of book is this?", "type": "text" } ], "role": "user" },
    { "content": [ { "index": null, "text": "Health, Fitness & Dieting", "type": "text" } ], "role": "assistant" },
    { "content": [ { "index": null, "text": "Is this a fitness book?", "type": "text" } ], "role": "user" },
    { "content": [ { "index": null, "text": "Yes", "type": "text" } ], "role": "assistant" }
]
```

