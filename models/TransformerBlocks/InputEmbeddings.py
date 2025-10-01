import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Converts token indices into dense embeddings scaled by sqrt(model_dimension).
    
    Why scaling?
    - Without scaling, the variance of embeddings may be too small compared to positional encodings.
    - Scaling ensures better-balanced inputs at the start of training (as suggested in the Transformer paper).
    """

    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):
        # Input: (batch_size, seq_len)
        # Output: (batch_size, seq_len, model_dimension)
        return self.embedding(x) * math.sqrt(self.model_dimension)