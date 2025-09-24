import torch.nn as nn

class ProjectionLayer(nn.Module):

    def __init__(self, model_dimension, vocab_size) -> None: 
        super().__init__()
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)