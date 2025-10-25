"""
This module implements the position-wise feed-forward network (FFN) that appears
in each Transformer layer after the attention mechanism.

Architecture:
-------------
    FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂

Where:
    - W₁: (d_model, d_ff) - expansion matrix
    - W₂: (d_ff, d_model) - contraction matrix
    - d_ff is typically 4× larger than d_model

Purpose:
--------
While attention captures relationships between positions, the FFN provides:
- Additional transformation capacity at each position
- Feature interaction within each token's representation
"""

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network used in Transformer encoder/decoder layers.
    
    Consists of two linear transformations with a ReLU activation and dropout in between.
    Applied identically to each position separately and identically.
    
    Attributes:
        linear_1 (nn.Linear): First linear layer (expansion)
        linear_2 (nn.Linear): Second linear layer (contraction)
        dropout (nn.Dropout): Dropout for regularization
    """

    def __init__(self, model_dimension: int, d_ff: int, dropout: float) -> None:
        """
        Initialize the Feed-Forward block.
        
        Args:
            model_dimension (int): Input and output dimension 
            d_ff (int): Inner/hidden dimension of FFN 
            dropout (float): Dropout probability applied after ReLU activation
        
        Architecture:
            Input → Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model) → Output
        """
        super().__init__()
        
        # First linear layer: expand from d_model to d_ff
        # This increases the representational capacity
        self.linear_1 = nn.Linear(model_dimension, d_ff)
        
        # Dropout for regularization (applied after activation)
        self.dropout = nn.Dropout(dropout)
        
        # Second linear layer: contract from d_ff back to d_model
        self.linear_2 = nn.Linear(d_ff, model_dimension)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dimension)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, model_dimension)
        
        Shape Transformation:
            (batch, seq_len, d_model) → (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        """
        # Step 1: Expand dimensions
        # (batch, seq_len, d_model) → (batch, seq_len, d_ff)
        x = self.linear_1(x)
        
        # Step 2: Apply ReLU activation
        x = torch.relu(x)
        
        # Step 3: Apply dropout for regularization
        x = self.dropout(x)
        
        # Step 4: Contract back to original dimension
        # (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        x = self.linear_2(x)
        
        return x