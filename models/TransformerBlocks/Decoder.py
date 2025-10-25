"""
This module implements the decoder component of the Transformer, which generates
the output sequence one token at a time, conditioned on the encoder's representations of the input.

Architecture Overview:
----------------------
Decoder = Stack of N identical DecoderBlocks

Each DecoderBlock contains:
1. Masked Multi-Head-Attention: attends only to previous positions
2. Multi-Head Cross-Attention: attends to encoder output
3. Feed-Forward Network: position-wise transformation
4. Residual connections + Layer Normalization around each sub-layer

Purpose:
--------
The decoder's job is to:
- Generate output sequence autoregressively (one token at a time)
- Attend to its own previous outputs (self-attention with masking)
- Attend to encoder representations (cross-attention)
- Predict the next token based on all available information

Why Three Sub-layers?
----------------------
1. **Masked Self-Attention**: "What have I generated so far?"
   - Allows decoder to build coherent sequences
   - Masked to prevent cheating during training

2. **Cross-Attention**: "What does the input say?"
   - Connects to encoder output

3. **Feed-Forward**: "Additional processing"
   - Non-linear transformations
   - Extra representational capacity

Information Flow:
-----------------
    Target Tokens (previous outputs)
        ↓
    Embedding + Positional Encoding
        ↓
    ┌──────────────────────────────┐
    │   Decoder Block 1            │
    │  - Masked Self-Attention     │
    │  - Cross-Attention           │
    │  - Feed-Forward              │
    └──────────────────────────────┘
        ↓                  
    ┌──────────────────────────────┐
    │   Decoder Block 2            │
    │       ...                    │
    └──────────────────────────────┘
        ↓                  
        ...                
        ↓                  
    ┌──────────────────────────────┐
    │   Decoder Block N            │
    └──────────────────────────────┘
        ↓                  
    Final Layer Norm       
        ↓                 
    Output Logits
"""

import torch.nn as nn

from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .ResidualConnection import ResidualConnection
from .FeedForwardBlock import FeedForwardBlock


class DecoderBlock(nn.Module):
    """
    Single decoder block - the fundamental building unit of the decoder.
    
    Each block has three main components:
    1. Masked self-attention (on target sequence)
    2. Cross-attention (to encoder output)
    3. Feed-forward network
    
    Architecture:
        Input
          ↓
        [Masked Self-Attention + Residual + Norm]
          ↓
        [Cross-Attention + Residual + Norm]
          ↓
        [Feed-Forward + Residual + Norm]
          ↓
        Output
    
    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Masked self-attention
        cross_attention_block (MultiHeadAttentionBlock): Encoder-decoder attention
        feed_forward_block (FeedForwardBlock): Position-wise FFN
        residual_connections (nn.ModuleList): Three residual connection modules
    """

    def __init__(
        self,
        model_dimension: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        """
        Initialize a decoder block.
        
        Args:
            model_dimension (int): Dimension of representations (d_model)
            self_attention_block (MultiHeadAttentionBlock): For masked self-attention
            cross_attention_block (MultiHeadAttentionBlock): For attending to encoder
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
            dropout (float): Dropout probability for residual connections
        """
        super().__init__()
        
        # 1. Masked self-attention: target tokens attend to previous target tokens
        self.self_attention_block = self_attention_block
        
        # 2. Cross-attention: target attends to encoder's source representations
        self.cross_attention_block = cross_attention_block
        
        # 3. Feed-forward network: additional non-linear transformation
        self.feed_forward_block = feed_forward_block
        
        # Three residual connections (one for each sub-layer)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(model_dimension, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through one decoder block.
        
        Args:
            x (torch.Tensor): Target sequence representations
                             Shape: (batch_size, tgt_seq_len, model_dimension)
            encoder_output (torch.Tensor): Encoder's output (source context)
                                          Shape: (batch_size, src_seq_len, model_dimension)
            src_mask (torch.Tensor): Source mask (ignore padding in encoder output)
                                    Shape: (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target mask (causal + padding)
                                    Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
                                    Prevents attending to future positions
        
        Returns:
            torch.Tensor: Transformed target representations
                         Shape: (batch_size, tgt_seq_len, model_dimension)
        """
        # Sub-layer 1: Masked Self-Attention with residual connection
        # Target attends to itself (past positions only due to tgt_mask)
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        
        # Sub-layer 2: Cross-Attention with residual connection
        # Target (query) attends to encoder output (key, value)
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        
        # Sub-layer 3: Feed-Forward Network with residual connection
        # Position-wise transformation applied independently
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x


class Decoder(nn.Module):
    """
    Full decoder: stack of N identical decoder blocks with final normalization.
    
    Attributes:
        layers (nn.ModuleList): List of N decoder blocks
        norm (nn.LayerNorm): Final layer normalization
    """

    def __init__(self, model_dimension: int, layers: nn.ModuleList) -> None:
        """
        Initialize the decoder with multiple stacked layers.
        
        Args:
            model_dimension (int): Dimension of representations (d_model)
            layers (nn.ModuleList): List of N DecoderBlock instances
        """
        super().__init__()
        
        # Stack of N decoder blocks
        self.layers = layers
        
        # Final layer normalization (applied after all blocks)
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the entire decoder.
        
        Args:
            x (torch.Tensor): Target embeddings (with positional encoding)
                             Shape: (batch_size, tgt_seq_len, model_dimension)
            encoder_output (torch.Tensor): Encoded source representations
                                          Shape: (batch_size, src_seq_len, model_dimension)
            src_mask (torch.Tensor): Source mask for padding
                                    Shape: (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target mask (causal + padding)
                                    Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            torch.Tensor: Decoded representations ready for projection
                         Shape: (batch_size, tgt_seq_len, model_dimension)
        """
        # Pass through all decoder blocks sequentially
        # Each block's output becomes the next block's input
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply final layer normalization
        return self.norm(x)