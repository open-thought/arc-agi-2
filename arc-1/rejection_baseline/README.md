# Rejection-Sampling ARC baseline


- sample completions for ARC riddles, measure perplexity of ground-truth outputs with given CoT
- select best N samples per riddle and fine-tune llm


Model: `meta-llama/Llama-3.2-3B-Instruct`


todo:
- [x] create train/eval split with 100 riddles for eval
- [x] shell script to start tgi with llama 3.2 3b locally
- [x] check how to get logprobs of prompt (prefill) 
- [x] measuring perplexity of existing dataset with tgi
- [x] sample parallel CoT completions with tgi
- [x] generate FT dataset
- [x] finetune on top-k examples



### Eval results (on 100 holdout riddles)

```

## stage 0 (original meta-llama/Llama-3.2-3B-Instruct)

pass@1: Avg ppl: 3.7717, ppl_count: 33, Solved: 0/100
pass@3: Avg ppl: 3.3644, ppl_count: 108, Solved: 0/100


## stage 1 (after 2nd epoch fine-tuned on best k=5 of 64 per riddle)

pass@1: Avg ppl: 3.6626, ppl_count: 69, Solved: 0/100
pass@3: Avg ppl: 3.4621, ppl_count: 254, Solved: 0/100

```


### debug axolotl tokenization

`python -m axolotl.cli.preprocess your_config.yml --debug`
