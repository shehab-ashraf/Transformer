import torch.nn as nn

from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .ResidualConnection import ResidualConnection
from .FeedForwardBlock import FeedForwardBlock

# ===========================
#  One Encoder Block
# ===========================
class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dimension: int,                    # Hidden/embedding dimension
        self_attention_block: MultiHeadAttentionBlock,  # Multi-head self-attention
        feed_forward_block: FeedForwardBlock,    # Position-wise feed-forward network
        dropout: float                           # Dropout rate
    ) -> None:
        super().__init__()
        # Multi-head self-attention module
        self.attention_block = self_attention_block
        # Feed-forward transformation module
        self.feed_forward_block = feed_forward_block
        # Two residual connections: 
        #   1. Around self-attention
        #   2. Around feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(model_dimension, dropout) for _ in range(2)
        ])
    
    def forward(self, x, src_mask):
        """
        Forward pass through one encoder block.
        x: input sequence embeddings (batch, seq_len, d_model)
        src_mask: mask for padding or attention control
        """
        # Step 1: Self-attention + residual connection + layer norm
        x = self.residual_connections[0](
            x,
            # Pass attention block as a function (query=key=value=x)
            lambda x: self.attention_block(x, x, x, src_mask)
        )
        # Step 2: Feed-forward network + residual connection + layer norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# ===========================
# Full Encoder (Stack of N EncoderBlocks)
# ===========================
class Encoder(nn.Module):
    def __init__(self, model_dimension: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers   # List of N encoder blocks
        # Final layer normalization (stabilizes training & output)
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x, mask):
        """
        Forward pass through the full encoder.
        x: input embeddings with positional encoding
        mask: source mask (to ignore padding tokens)
        """
        # Pass input sequentially through all encoder blocks
        for layer in self.layers: 
            x = layer(x, mask)
        # Apply final layer normalization
        return self.norm(x)
