"""
This module implements the encoder component of the Transformer, which processes
the source sequence into contextualized representations that capture relationships between all input tokens.

Architecture Overview:
----------------------
Encoder = Stack of N identical EncoderBlocks

Each EncoderBlock contains:
1. Multi-Head-Attention: tokens attend to all other tokens
2. Feed-Forward Network: position-wise transformation
3. Residual connections + Layer Normalization around each sub-layer

Purpose:
--------
The encoder's job is to:
- Read and understand the input sequence
- Build rich contextualized representations
- Allow each token to gather information from entire sequence

Information Flow:
-----------------
    Input Tokens
        ↓
    Embedding + Positional Encoding
        ↓
    ┌─────────────────────────┐
    │   Encoder Block 1       │
    │  - Multi-Head-Attention │
    │  - Feed-Forward         │
    └─────────────────────────┘
        ↓
    ┌─────────────────────────┐
    │   Encoder Block 2       │
    │       ...               │
    └─────────────────────────┘
        ↓
        ...
        ↓
    ┌─────────────────────────┐
    │   Encoder Block N       │
    └─────────────────────────┘
        ↓
    Final Layer Norm
        ↓
    Contextualized Representations
"""

import torch.nn as nn

from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .ResidualConnection import ResidualConnection
from .FeedForwardBlock import FeedForwardBlock


class EncoderBlock(nn.Module):
    """
    Single encoder block - the fundamental building unit of the encoder.
    
    Each block transforms the input through Multi-Head-Attention and feed-forward
    layers, with residual connections and layer normalization for stable training.
    
    Architecture:
        Input
          ↓
        [Multi-Head-Attention + Residual + Norm]
          ↓
        [Feed-Forward + Residual + Norm]
          ↓
        Output
    """

    def __init__(
        self,
        model_dimension: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        """
        Initialize an encoder block.
        
        Args:
            model_dimension (int): Dimension of representations (d_model)
            self_attention_block (MultiHeadAttentionBlock): Multi-head attention module
            feed_forward_block (FeedForwardBlock): Feed-forward network module
            dropout (float): Dropout probability for residual connections
        
        Structure:
            Two main sub-layers, each wrapped with residual connection:
            1. Multi-head-attention
            2. feed-forward network
        """
        super().__init__()
        
        # Multi-head-attention: allows tokens to attend to each other
        self.attention_block = self_attention_block
        
        # Feed-forward network: additional non-linear transformation
        self.feed_forward_block = feed_forward_block
        
        # Two residual connections (one for attention, one for FFN)
        # Each includes layer norm, dropout, and skip connection
        self.residual_connections = nn.ModuleList([
            ResidualConnection(model_dimension, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        """
        Forward pass through one encoder block.
        
        Args:
            x (torch.Tensor): Input sequence representations
                             Shape: (batch_size, seq_len, model_dimension)
            src_mask (torch.Tensor): Mask for source sequence
                                    Shape: (batch_size, 1, 1, seq_len)
                                    Used to ignore padding tokens
        
        Returns:
            torch.Tensor: Transformed representations
                         Shape: (batch_size, seq_len, model_dimension)
    
        """
        # Sub-layer 1: Multi-Head-Attention with residual connection
        x = self.residual_connections[0](
            x,
            lambda x: self.attention_block(x, x, x, src_mask)
        )
        
        # Sub-layer 2: Feed-Forward Network with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x


class Encoder(nn.Module):
    """
    Full encoder: stack of N identical encoder blocks with final normalization.
    
    Attributes:
        layers (nn.ModuleList): List of N encoder blocks
        norm (nn.LayerNorm): Final layer normalization
    """

    def __init__(self, model_dimension: int, layers: nn.ModuleList) -> None:
        """
        Initialize the encoder with multiple stacked layers.
        
        Args:
            model_dimension (int): Dimension of representations (d_model)
            layers (nn.ModuleList): List of N EncoderBlock instances
        """
        super().__init__()
        
        # Stack of N encoder blocks
        self.layers = layers
        
        # Final layer normalization (applied after all blocks)
        # Stabilizes the output and helps with training
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x, mask):
        """
        Forward pass through the entire encoder.
        
        Args:
            x (torch.Tensor): Input embeddings (with positional encoding)
                              Shape: (batch_size, src_seq_len, model_dimension)
            mask (torch.Tensor): Source mask to ignore padding tokens
                                 Shape: (batch_size, 1, 1, src_seq_len)
        
        Returns:
            torch.Tensor: Encoded representations
                          Shape: (batch_size, src_seq_len, model_dimension)
        """
        # Pass through all encoder blocks sequentially
        # Each block's output becomes the next block's input
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        # This ensures the encoder output has stable statistics
        return self.norm(x)