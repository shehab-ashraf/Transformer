import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used in Transformer layers.

    Each token is processed independently through:
    1. Linear layer: expands from model_dimension → d_ff (usually 4x bigger).
    2. ReLU activation: introduces non-linearity.
    3. Dropout: adds regularization.
    4. Linear layer: projects back from d_ff → model_dimension.

    This block gives the model extra expressive power by enabling 
    non-linear transformations of each token’s representation.
    """

    def __init__(self, model_dimension: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(model_dimension, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, model_dimension) 

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
