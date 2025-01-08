# Rejection-Sampling ARC baseline


- sample completions for ARC riddles, measure perplexity of ground-truth outputs with given CoT
- select best N samples per riddle and fine-tune llm


Model: `meta-llama/Llama-3.2-3B-Instruct`


todo:
- [ ] create train/eval split with 100 riddles for eval

- [x] shell script to start tgi with llama 3.2 3b locally
- [x] check how to get logprobs of prompt (prefill) 
- [ ] measuring perplexity of existing dataset with tgi
- [ ] sample parallel CoT completions with tgi
- [ ] generate FT dataset
- [ ] finetune on examples

