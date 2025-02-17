# ARC Brainstorming/Ideas (Dec 2024)

## Classic transduction approach

- motivation: use TTT (program search/hypotheses filtering via gradient descent)
- data is king: scale high-quality synthetic riddle generation
	- aim for collisions with the private ARC test-set, most ARC-AGI-1 riddles were probably generated by Chollet himself, attack vector: "replicate Chollet" - already extracted the "[core-ideas](https://github.com/open-thought/arc-agi-2/blob/main/arc-1/annotated-re-arc/core_idea_samples.json)" behind ARC-1 with Sonnet and generated >1k derived riddle-generator+verifier pairs (open-ended generation starting from core-idea descriptions and re-arc generators as seed-samples).
	- maximize synthetic riddle diversity and quality, e.g. by selecting in-context examples for generation cleverly (take inspiration from [Novelty Search](https://algorithmafternoon.com/novelty/novelty_search_algorithm/)) and filtering poor outputs or near-duplicates
	- strive for new original ideas, e.g. use human in the loop for steering & filtering/ranking of synthetic riddle generation
	- collect input & output of human feedback to capture the human-preferences (e.g. to filter out-of-human-distribution samples out)
- search best open-weights model for transduction fine-tuning (probably needs to be <= 8B), test coding and reasoning-models
- catch up with experienced teams: gain experience how transduction generalizes (e.g. by training data of single re-arc generator or simple riddle sub-sets, e.g. generators with lowest number of source-lines)
- try fine-tuning a VLM, e.g. visual representation ARC boards (vision encoders might have advantages in recognizing 2D spatial relations)
- collect a list of riddle board augmentation techniques and test their effectiveness for fine-tuning and during TTT (e.g. see [arcmentations](https://github.com/arc-community/arcmentations))
- conduct TTT experiments: augmentations, freezing layers, training only FFN/KQV, different optimizers, regularization & hyper parameters
- check effectiveness of hallucination detection (via internal state classifier) and mitigation for ARC riddles (e.g. paper [FactCheckmate](https://arxiv.org/abs/2410.02899))
- use the test-input board for consistency-checking: predict test output & swap with training example -> verify that training example output is predicted correctly given test-input&output as training pair (check if this consistency-test can be used in the TTT objective).
- try to apply nanoGPT speedrun ([leaderboard](https://app.primeintellect.ai/speedrun/nanogpt)) insights (e.g. [Muon optimizer](https://github.com/KellerJordan/Muon))
- evaluate effectiveness of adding `<pause>` thought-tokens during fine-tuning ([paper](https://arxiv.org/abs/2310.02226) and [coconut](https://arxiv.org/abs/2412.06769))
- train board transformation from natural language description task (separate the deduction of the board transformation from its execution)


## ARC inductive coding agent

- motivation: human interpretable program search heuristic with intermediate results (potentially multi-step)
- create an ARC agent environment which allows riddle analysis, solution checking, plan storage and code execution, e.g. to  alliteratively re-writing improve & fix synthesized transformation-program candidates
	- find good structured prompt template containing the training examples, plan/idea, history of failed attempts, found solutions and DSL functions used for "similar" riddles (retrieved via RAG), current program candidate, resulting output board or compiler/interpreter error messages, etc.
- generate an ARC CoT "reasoning" dataset of transformation description based on synthetic riddle-generator source-code as oracles
	- describe the content and features of the training boards, commonalities & differences, obvious relationship between input and output
- train a model to generate transformation hypotheses (natural language)
- test effectiveness of RAG: train a model to describe the first impression or core-concept of a riddle. Use embeddings of description for retrieval of similar solutions in riddle db.
- train verifier / process reward model & use tree-search-techniques to find correct result: detect invalid a hypothesis as soon as possible & backtrack (before synthesizing a candidate program)
- reflect with a strong teacher model about failures -> generate data for back-tracking/self-correction, give oracle knowledge about riddle-generators to teacher model
- augment riddle with basic data of riddle boards (in/out sizes, color-histogram, segmentation)
- use meta-planning prompts to summarize and criticize a rollout and to make suggestions what should be tried next (potentially combine with RL to learn what an effective strategy for ARC riddles is, e.g. what to ask/try first, which prompts/questions uncover the transformation rules)

### Agent implementation
- training data: generate riddle board descriptions with the riddle-generator source as oracel information (not shown during training)
- optimize key prompts of system in outer loop with llm (learn to ask the 'right questions' in higher meta-cognition layer)
- reflect on failed attempts with llm to generate filtered/summarized context for future request
- use llm to generate corrected reasoning traces from failed attempts when shown ground-truth example
- adaptive sampling: start sampling with low temperature and increase temperature gradually for reasoning (requires solid verifier)
- aim to limit the amount of reasoning-work: for simple 'well-known' riddles a direct transduction should be enough, try quick-shots first and check them before starting more elaborate reasoning (requires solid verifier, i.e. simpler for induction/program synthesis)
- try to classify based on a generated riddle-analysis if an induction vs. transduction approach should be tried first


## Combine induction & transduction

- challenge: llm program generation is slow
- estimate max llm based program generation throughput (e.g. non-naive tree-search with prefix-caching)
