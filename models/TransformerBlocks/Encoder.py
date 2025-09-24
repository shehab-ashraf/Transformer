import torch.nn as nn

from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .ResidualConnection import ResidualConnection
from .FeedForwardBlock import FeedForwardBlock

class EncoderBlock(nn.Module):

    def __init__(self, model_dimension: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(model_dimension, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, model_dimension: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x, mask):
        for layer in self.layers: 
            x = layer(x, mask)
        return self.norm(x)