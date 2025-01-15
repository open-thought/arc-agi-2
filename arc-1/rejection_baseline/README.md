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

## stage 2 
pass@1: Avg ppl: 18.43298559619863, ppl_count: 74, Solved: 0/100 (avg heavily skewed by outliers with high ppl)
pass@3: Avg ppl: 6.070093191589242, ppl_count: 251, Solved: 1/100, IDs of solved riddles:  ['ff28f65a']
```

Sampling

```
## stage 1: output/llama-3.2-3B-instruct-stage1-simple-t5-16.jsonl
sampling Avg ppl: 8.887493558618504, ppl_count: 8537

Solved: 18/700
Skipped: 0
IDs of solved riddles:  ['332efdb3', '3b4c2228', '445eab21', '5582e5ca', '68b67ca3', '6f8cd79b', '74dd1130', '794b24be', '8597cfd7', '9110e3c5', '963e52fc', 'a416b8f3', 'c3f564a4', 'c8b7cc0f', 'd037b0a7', 'd10ecb37', 'd5c634a2', 'fc754716']

```


### debug axolotl tokenization

`python -m axolotl.cli.preprocess your_config.yml --debug`
