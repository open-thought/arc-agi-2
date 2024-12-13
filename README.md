# arc-agi-2

## Repository Index

- [ARC-AGI Material Link List](./docs/research.md)
- [tgi-eval](./arc-1/tgi-eval/) - using ARC-AGI-1 to evaluate LLMs
- [Annotated RE-ARC](./arc-1/annotated-re-arc/README.md)


## Building the cognitive-core to solve ARC-AGI-2

2024-12-10 [andreaskoepf](https://x.com/neurosp1ke/)

Welcome fellow AI enthusiast!

You have probably heard about François Chollet's [ARC-Price](https://arcprize.org/) competition. The 2024 ARC-Price is now over and it is time to prepare for next year's ARC-AGI-2. I therefore started the **OpenThough ARC-AGI team**.

Our mission is to **develop a cognitive core to solve ARC-AGI-2** 🙂 - and to do this as much as possible open-source.

My preliminary plan is to follow a transformer based transduction approach, which involves:

- generating high quality synthetic data (like my riddle-generator+verifier pairs: https://x.com/neurosp1ke/status/1854796093801775215 )
- fine-tuning best open-weight LLM/VLMs
- training multi-turn reasoning (meta-cognition, planning, process-feedback-models, reflection, correction/back-tracking, etc.)
- applying test-time fine-tuning

Any help building the team and project is of course welcome. I will try to break down the project into many small work chunks which can be developed open-source (also to help other teams). The idea is to allow collaboration even without long-term commitment.

If you have feedback or want to chat about reasoning models, self-improvement, test-time fine-tuning or ARC-AGI in general, don't hesitate to contact me in the `#arc-agi-2` channel on [GPU mode](https://discord.gg/gpumode).
