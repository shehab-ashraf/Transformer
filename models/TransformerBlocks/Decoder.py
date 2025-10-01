import torch.nn as nn

from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .ResidualConnection import ResidualConnection
from .FeedForwardBlock import FeedForwardBlock

# ===========================
#    One Decoder Block
# ===========================
class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dimension: int,                        # embedding dimension
        self_attention_block: MultiHeadAttentionBlock,  # Masked self-attention
        cross_attention_block: MultiHeadAttentionBlock, # Encoder–decoder attention
        feed_forward_block: FeedForwardBlock,        # Position-wise feed-forward
        dropout: float                               # Dropout rate
    ) -> None: 
        super().__init__()
        # 1. Masked self-attention (target attends to itself, no looking ahead)
        self.self_attention_block = self_attention_block
        # 2. Cross-attention (target attends to encoder’s output for source context)
        self.cross_attention_block = cross_attention_block
        # 3. Feed-forward transformation
        self.feed_forward_block = feed_forward_block
        # Three residual connections: one for each sub-layer
        self.residual_connections = nn.ModuleList([
            ResidualConnection(model_dimension, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through one decoder block.
        x: target embeddings/representations (batch, tgt_seq_len, d_model)
        encoder_output: encoder’s final representations (batch, src_seq_len, d_model)
        src_mask: mask to ignore source padding tokens
        tgt_mask: causal mask + target padding mask
        """
        # Step 1: Masked self-attention (target tokens attend only to past tokens)
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # Step 2: Cross-attention (attend to encoder’s outputs)
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        # Step 3: Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


# ===========================
#  Full Decoder (Stack of N DecoderBlocks)
# ===========================
class Decoder(nn.Module):
    def __init__(self, model_dimension: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers   # List of N decoder blocks
        # Final layer normalization (applied after all layers)
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the full decoder.
        x: target embeddings with positional encoding
        encoder_output: source sequence representations from encoder
        src_mask: source mask (ignore padding in source)
        tgt_mask: target mask (causal + padding)
        """
        # Pass through all decoder blocks sequentially
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply final layer normalization
        return self.norm(x)
