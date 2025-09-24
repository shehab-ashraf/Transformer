import torch.nn as nn

from .InputEmbeddings import InputEmbeddings
from .PositionalEncoding import PositionalEncoding
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .FeedForwardBlock import FeedForwardBlock
from .ProjectionLayer import ProjectionLayer
from .Encoder import EncoderBlock, Encoder
from .Decoder import DecoderBlock, Decoder

class Transformer(nn.Module):


    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, model_dimension: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> None:
        super().__init__()


        # Encoder part
        src_embed = InputEmbeddings(model_dimension, src_vocab_size)
        src_pos = PositionalEncoding(model_dimension, src_seq_len, dropout)
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
            feed_forward_block = FeedForwardBlock(model_dimension, d_ff, dropout)
            encoder_block = EncoderBlock(model_dimension, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))

        # Decoder part 
        tgt_embed = InputEmbeddings(model_dimension, tgt_vocab_size)
        tgt_pos = PositionalEncoding(model_dimension, tgt_seq_len, dropout)
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
            feed_forward_block = FeedForwardBlock(model_dimension, d_ff, dropout)
            decoder_block = DecoderBlock(model_dimension, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)
        decoder = Decoder(model_dimension, nn.ModuleList(decoder_blocks))


        # projection layer
        projection_layer = ProjectionLayer(model_dimension, tgt_vocab_size)

        self.encoder = encoder
        self.decoder = decoder  
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed 
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


        # xavier initialization
        self.init_with_xavier()

        # Sharing weights between two embedding layers
        self.src_embed.weight = self.tgt_embed.weight


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def init_with_xavier(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)