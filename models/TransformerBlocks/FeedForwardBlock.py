import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):

    def __init__(self, model_dimension: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(model_dimension, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, model_dimension) 

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))