import torch.nn as nn

from .InputEmbeddings import InputEmbeddings
from .PositionalEncoding import PositionalEncoding
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .FeedForwardBlock import FeedForwardBlock
from .ProjectionLayer import ProjectionLayer
from .Encoder import EncoderBlock, Encoder
from .Decoder import DecoderBlock, Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,   # Size of source vocabulary
        tgt_vocab_size: int,   # Size of target vocabulary
        src_seq_len: int,      # Max length of source sequence
        tgt_seq_len: int,      # Max length of target sequence
        model_dimension: int=512, # Embedding/hidden dimension
        N: int=6,              # Number of encoder/decoder layers
        h: int=8,              # Number of attention heads
        dropout: float=0.1,    # Dropout rate
        d_ff: int=2048         # Hidden size of feed-forward network
    ) -> None:
        super().__init__()

        # ============ ENCODER ============ #
        # Convert token indices → dense vectors
        src_embed = InputEmbeddings(model_dimension, src_vocab_size)
        # Add positional information to embeddings
        src_pos = PositionalEncoding(model_dimension, src_seq_len, dropout)

        # Stack N identical encoder blocks
        encoder_blocks = []
        for _ in range(N):
            # Self-attention: tokens attend to each other in the same sequence
            encoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
            # Feed-forward network: local nonlinear transformation
            feed_forward_block = FeedForwardBlock(model_dimension, d_ff, dropout)
            # One complete encoder block (Attention + FFN + Residual + Norm)
            encoder_block = EncoderBlock(model_dimension, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        
        encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))


        # ============ DECODER ============ #
        # Convert target token indices → dense vectors
        tgt_embed = InputEmbeddings(model_dimension, tgt_vocab_size)
        # Add positional information
        tgt_pos = PositionalEncoding(model_dimension, tgt_seq_len, dropout)

        # Stack N identical decoder blocks
        decoder_blocks = []
        for _ in range(N):
            # 1. Self-attention (masked): each target token attends only to past tokens
            decoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
            # 2. Cross-attention: attends to encoder’s output (source context)
            decoder_cross_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
            # 3. Feed-forward network
            feed_forward_block = FeedForwardBlock(model_dimension, d_ff, dropout)
            # One complete decoder block (Self-Attn + Cross-Attn + FFN + Residual + Norm)
            decoder_block = DecoderBlock(model_dimension, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)
        
        decoder = Decoder(model_dimension, nn.ModuleList(decoder_blocks))


        # ============ FINAL PROJECTION ============ #
        # Project decoder outputs → target vocabulary logits
        projection_layer = ProjectionLayer(model_dimension, tgt_vocab_size)


        # Save components as class attributes
        self.encoder = encoder
        self.decoder = decoder  
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed 
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


        # ============ WEIGHT SHARING ============ #
        # Share parameters between:
        #   - Source & target embedding layers
        #   - Target embedding & final projection layer
        self.src_embed.embedding.weight = self.tgt_embed.embedding.weight
        self.projection_layer.proj.weight = self.tgt_embed.embedding.weight


        # ============ INITIALIZATION ============ #
        # Apply Xavier initialization to all weight matrices
        self.init_with_xavier()


    # Forward pass utilities
    def encode(self, src, src_mask):
        """Encode source sequence into context representations."""
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        """Decode target sequence using encoder’s memory + masks."""
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def project(self, x):
        """Project decoder outputs into target vocabulary logits."""
        return self.projection_layer(x)
    
    def init_with_xavier(self):
        """Apply Xavier initialization to all multi-dimensional weights."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
