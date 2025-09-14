import torch 
import torch.nn as nn

import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        return self.embedding(x) * torch.sqrt(self.d_model) # (stated in the paper) scale by sqrt(d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position_id = torch.arange(0, max_seq_len).unsqueeze(1) # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(-math.log(10000.0) / d_model))

        positional_encodings_table = torch.zeros(max_seq_len, d_model)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * div_term)
        positional_encodings_table[:, 1::2] = torch.cos(position_id * div_term)

        self.register_buffer('positionl_encodings_table', positional_encodings_table)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {x.shape}'
        
        positional_encodings = self.positional_encodings_table[:x.shape[1]]
        return self.dropout(x + positional_encodings)