"""
This module implements residual (skip) connections with layer normalization,
a crucial architectural component that enables training of deep networks.

Key Concepts:
-------------
1. **Residual Connection**: Adds input directly to output of a layer
2. **Layer Normalization**: Normalizes across feature dimension for stability
3. **Pre-LN vs Post-LN**: This uses Pre-LN (normalize before sublayer)

Why Residual Connections?
--------------------------
- Enable gradient flow through deep networks (solves vanishing gradients)
- Allow network to learn identity function easily (if needed)
- Provide "gradient highways" for efficient backpropagation
- Essential for training models with 6+ layers

Pre-LN Architecture:
--------------------
    output = x + Dropout(Sublayer(LayerNorm(x)))

Original Transformer used Post-LN:
    output = LayerNorm(x + Dropout(Sublayer(x)))

Pre-LN is now preferred because:
- More stable training (especially for deep models)
- Less sensitive to learning rate
- Better gradient flow
"""

import torch.nn as nn


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        """
        
        Args:
            features (int): Number of features (d_model, typically 512)
                          This is the dimension across which normalization is applied
            dropout (float): Dropout probability (typically 0.1)
        
        Components:
            - LayerNorm: Normalizes features to have mean=0, std=1
            - Dropout: Randomly zeros elements for regularization
        """
        super().__init__()
        
        # Layer normalization: normalizes across the feature dimension
        self.norm = nn.LayerNorm(features)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection around a sublayer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features)
            sublayer (callable): Function/module to apply (attention or FFN)
        
        Returns:
            torch.Tensor: Output with residual connection (batch_size, seq_len, features)
        
        Process (Pre-LN):
            1. Normalize the input (stabilizes training)
            2. Apply the sublayer transformation
            3. Apply dropout to the sublayer output
            4. Add the original input (residual connection)
        
        Why this order?
            Pre-LN (norm → sublayer → add) is more stable than
            Post-LN (sublayer → add → norm) for deep networks.
        
        Shape Preservation:
            All operations maintain the same tensor shape, which is
            critical for the residual addition to work correctly.
        """
        # Step 1: Apply layer normalization to input
        # Normalizes across the feature dimension (last dim)
        normalized = self.norm(x)
        
        # Step 2: Apply the sublayer (attention or feed-forward)
        # The sublayer receives normalized input
        sublayer_output = sublayer(normalized)
        
        # Step 3: Apply dropout for regularization
        # Randomly zeros some elements of the sublayer output
        dropped = self.dropout(sublayer_output)
        
        # Step 4: Add residual connection (skip connection)
        # This allows gradients to flow directly through the network
        # Shape: (batch, seq_len, features) + (batch, seq_len, features)
        return x + dropped