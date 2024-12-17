# Plan

## Main Goals
- [ ] SFT multi-modal ARC reasoning model
- [ ] RL multi-modal ARC reasoning model

## Evaluation
- [ ] text
- [ ] text+CoT
- [ ] image+text
- [ ] image+text+CoT
- [ ] text+CoT scored by PRM
- [ ] text+image+CoT scored by PRM
- [ ] text+image+CoT+RL scored by PRM

## TODO
- [ ] create initial dataset
    - [ ] transduction text
    - [ ] transduction image+text
- [ ] generate a synthetic dataset (min 10k samples) with CoT reasoning for ARC riddles
    - [ ] text-only input
    - [ ] image+text input
- [ ] train llama 3.2 vision 11B for ARC transduction task
    - [ ] test & decide Unsloth vs. Axolotl
    - [ ] determine compute requirements
    - [ ] run first FT
- [ ] generate reward verifier dataset
    - [ ] generate dataset with pos+neg examples of riddle analysis
    - [ ] generate dataset with pos+neg examples of board transformations
- [ ] train outcome reward model and check if it is suitable as PRM
- [ ] setup RL training against hard verifier (synthetic augmented riddles)

### Extensions
- Generate dataset with additional tasks
    - [ ] task: decoding of boards image to text representations (using different color maps)
    - [ ] task: transforming a single input board based on a transformation description
    - [ ] task: induction / program generation
