import argparse

# Llama 3.2-Vision 11B, 128k context
# native image tile size of vision adapter: 448 px

# Llama 3.2-Vision: https://huggingface.co/docs/transformers/main/model_doc/mllama
# Unsloth llama vision finetuning notebook: https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing
# Axolotl example: https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml


# ask Sonnet to generate an analyse and instruction to for an ARC riddle based on generator code
# 1. Analysis of the riddle, first impression, riddle class and core idea, concepts used objects
# 2. Detailed natural language program which explains the transformation


# Ask Sonnet to verify and possibly improve its own description when shown without the corresponding generator/verifier code.
# try generation with image input


# 1. load riddle examples + generator & verifier
# 2. create text representation of riddle
# 3. generate riddle visualization representation png image
#    - generate single image with overview of transformation
#    - generate multiple (specify color palette, optionally write numbers into fields)
# prepare data and send prompt to sonnet


def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    return args


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
