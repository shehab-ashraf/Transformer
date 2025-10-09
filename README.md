# Transformer
This repository contains a complete and well-documented PyTorch implementation of the Transformer model from the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. The project focuses on machine translation, efficient training with PyTorch Lightning, and experiment tracking with Weights & Biases. It’s designed to be a clear and practical resource for understanding and building Transformer models from scratch.

## Table of Contents
- [Architecture](#architecture)


## Architecture
<img src="/images/transformer.excalidraw.svg"  width="800">

The Transformer is based on the encoder-decoder architecture, where a sequence of words is translated from one language to another.
This architecture consists of two components:

- **Encoder**: Converts an input sequence of tokens into a sequence of embedding vectors(contextualized representations).
- **Decoder**: Uses the encoder’s output to iteratively generate an output sequence of tokens, one token at a time.