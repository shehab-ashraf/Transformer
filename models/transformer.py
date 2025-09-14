import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):
        # (batch_size, seq_len) --> (batch_size, seq_len, model_dimension)
        return self.embedding(x) * torch.sqrt(self.model_dimension) # (stated in the paper) scale by sqrt(model_dimension)
    
class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position_id = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2) * -(-math.log(10000.0) / model_dimension))

        positional_encodings_table = torch.zeros(max_seq_len, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * div_term)
        positional_encodings_table[:, 1::2] = torch.cos(position_id * div_term)

        self.register_buffer('positionl_encodings_table', positional_encodings_table)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {x.shape}'
        
        positional_encodings = self.positional_encodings_table[:x.shape[1]]
        return self.dropout(x + positional_encodings)
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, model_dimension: int, number_of_heads: int, dropout: float) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = model_dimension // number_of_heads
        self.w_q = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_k = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_v = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_o = nn.Linear(model_dimension, model_dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key , value, mask, dropout: nn.Dropout):
        head_dimension = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)

        if mask is not None: 
            attention_scores.maked_fill_(mask == 0, -1e9)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.head_dimension).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.head_dimension)

        return self.w_o(x)
    

class FeedForwardBlock(nn.Module):

    def __init__(self, model_dimension: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(model_dimension, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, model_dimension) 

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layres: 
            x = layer(x)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None: 
        super().__init__()
        self.self_attention_block = self_attention_block
        self.corss_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layres = layers
        self.norm = nn.LayerNorm(features)

    def forwrad(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layres:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    


class ProjectionLayer(nn.Module):

    def __init__(self, model_dimension, vocab_size) -> None: 
        super().__init__()
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, model_dimension: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) :
    
    # create the embedding layers
    src_embed = InputEmbeddings(model_dimension, src_vocab_size)
    tgt_embed = InputEmbeddings(model_dimension, tgt_vocab_size)

    # create the positonal encoding layers
    src_pos = PositionalEncoding(model_dimension, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(model_dimension, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, d_ff, dropout)
        encoder_block = EncoderBlock(model_dimension, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(model_dimension, h, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, d_ff, dropout)
        decoder_block = DecoderBlock(model_dimension, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # create the encoder and decoder
    encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))
    decoder = Decoder(model_dimension, nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(model_dimension, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

class Transformer(nn.Module):


    def __init__(self,  endcoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = endcoder
        self.decoder = decoder  
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed 
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def project(self, x):
        return self.project(x)
