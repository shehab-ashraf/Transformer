import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position_id = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2) * (-math.log(10000.0) / model_dimension)
        )

        positional_encodings_table = torch.zeros(max_seq_len, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * div_term)
        positional_encodings_table[:, 1::2] = torch.cos(position_id * div_term)

        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {x.shape}'
        
        positional_encodings = self.positional_encodings_table[:x.shape[1]]
        return self.dropout(x + positional_encodings)