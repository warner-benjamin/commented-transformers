# Commented Transformers

Highly commented implementations of Transformers in PyTorch for *Creating a Transformer From Scratch* series:

1. [The Attention Mechanism](https://benjaminwarner.dev/2023/07/01/attention-mechanism.html)
2. [The Rest of the Transformer](https://benjaminwarner.dev/2023/07/28/rest-of-the-transformer.html)


The layers folder contains implementations for `BidirectionalAttention`, `CausalAttention`, and `CausalCrossAttention`.

The models folder contains implementations for `GPT` and `BERT`. Both models are compatible with `torch.compile(..., fullgraph=True)`.