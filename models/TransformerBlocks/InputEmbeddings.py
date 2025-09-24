import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):
        # (batch_size, seq_len) --> (batch_size, seq_len, model_dimension)
        return self.embedding(x) * math.sqrt(self.model_dimension) # (stated in the paper) scale by sqrt(model_dimension)
    