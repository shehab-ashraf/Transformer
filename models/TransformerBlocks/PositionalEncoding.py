import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to embeddings.
    
    Idea:
    - Each position in the sequence gets a unique pattern of sines & cosines.
    - Even indices use sine, odd indices use cosine.
    - This encodes both absolute position and relative distance.
    """

    def __init__(self, model_dimension: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        position_id = torch.arange(0, max_seq_len).unsqueeze(1)

        # Divisor term for different frequencies
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2) * (-math.log(10000.0) / model_dimension)
        )

        # Precompute sinusoidal table
        positional_encodings_table = torch.zeros(max_seq_len, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * div_term)  # even dimensions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * div_term)  # odd dimensions

        # Register as buffer so it's not a parameter (not updated by gradients)
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, model_dimension)
        Returns:
            Embeddings enriched with positional information
        """
        # Slice encodings to match sequence length
        positional_encodings = self.positional_encodings_table[:x.shape[1]]
        
        # Add positions to embeddings and apply dropout
        return self.dropout(x + positional_encodings)