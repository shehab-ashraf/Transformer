"""
This module implements the final output layer of the Transformer decoder,
projecting hidden representations into vocabulary space to generate predictions.

Key Concepts:
-------------
1. **Vocabulary Projection**: Maps decoder outputs to vocabulary logits
2. **Linear Transformation**: Simple matrix multiplication (no activation)
3. **Logits**: Raw scores for each vocabulary token (before softmax)

Purpose:
--------
The decoder produces continuous hidden representations (vectors in d_model space).
To generate actual tokens, we need to:
- Convert these representations to scores over the vocabulary
- Each position gets a score for every possible token

Mathematical Operation:
-----------------------
    logits = x · W^T
    
Where:
    - x: decoder output (batch, seq_len, d_model)
    - W: weight matrix (vocab_size, d_model)
    - logits: vocabulary scores (batch, seq_len, vocab_size)

Weight Sharing:
---------------
Often, the projection weight matrix is shared (tied) with the target
embedding matrix. This:
- Reduces parameters
- Enforces consistency: similar embeddings → similar predictions
- Improves generalization
"""

import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Final projection layer that converts decoder outputs to vocabulary logits.
    
    This layer transforms the continuous hidden representations from the decoder
    into discrete token predictions by projecting to vocabulary space.
    
    Attributes:
        proj (nn.Linear): Linear projection layer mapping d_model → vocab_size
    """

    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        """
        Initialize the projection layer.
        
        Args:
            model_dimension (int): Size of hidden representations (d_model, typically 512)
            vocab_size (int): Size of target vocabulary (number of possible output tokens)
        """
        super().__init__()
        
        # Linear projection: d_model → vocab_size
        # Projects each position's representation to vocabulary space
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x):
        """
        Project decoder representations to vocabulary logits.
        
        Args:
            x (torch.Tensor): Decoder output representations
                             Shape: (batch_size, seq_len, model_dimension)
        
        Returns:
            torch.Tensor: Vocabulary logits (unnormalized scores)
                         Shape: (batch_size, seq_len, vocab_size)
        """
        # Apply linear projection
        # Shape: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        return self.proj(x)