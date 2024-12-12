# Annotated RE-ARC

## Extracting the Core Idea of ARC-AGI-1 Riddles

This folder contains the results of an experiment to generate textual descriptions of the core-ideas of all ARC-AGI-1 riddles with an LLM. The hand-crafted generators and verifiers of Michael Hodel's [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) were first LLM-augmented with line-by-line source comments and then passed as reference in the prompt to generate the the core idea descriptions.

The core-ideas (together with the annotated generators and verifiers) can for example be used as seed examples for open-ended LLM based synthetic riddle generation.


## Approach

1. Add comments to each generator function in [generators.py](https://github.com/michaelhodel/re-arc/blob/79e402eea8623191880f51238f36df393b2500a3/generators.py).
2. Add comments to each verifier function in [verifiers.py](https://github.com/michaelhodel/re-arc/blob/79e402eea8623191880f51238f36df393b2500a3/verifiers.py) using the commented generator as additional reference.
3. Take the annotated generator and verifier pair for each riddle and generate a description of the core-idea.


## Results

- [annotated_generators.json](./annotated_generators.json)
- [annotated_verifiers.json](./annotated_generators.json)
- [core_idea_samples.json](./core_idea_samples.json) (4 samples per riddle were generated)
