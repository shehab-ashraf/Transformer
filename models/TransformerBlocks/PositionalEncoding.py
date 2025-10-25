"""
This module adds positional information to token embeddings using sinusoidal functions.
Since Transformers don't have inherent sequential ordering (unlike RNNs), positional
encodings inject information about token positions in the sequence.

Key Concepts:
-------------
1. Uses sine and cosine functions of different frequencies
2. Even dimensions use sine, odd dimensions use cosine
3. Allows model to learn relative positions (tokens can "see" distance between positions)
4. Deterministic (not learned) - same position always gets same encoding
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to input embeddings.
    
    This module creates a fixed positional encoding matrix and adds it to the
    input embeddings, allowing the model to utilize sequence order information.
    
    Attributes:
        dropout (nn.Dropout): Dropout layer for regularization
        positional_encodings_table (torch.Tensor): Precomputed encoding matrix
                                                    Shape: (max_seq_len, model_dimension)
    """

    def __init__(self, model_dimension: int, max_seq_len: int, dropout: float) -> None:
        """
        Initialize positional encoding with precomputed sinusoidal patterns.
        
        Args:
            model_dimension (int): Size of embedding vectors (must match input embeddings)
            max_seq_len (int): Maximum sequence length to support
            dropout (float): Dropout probability applied after adding positions
        
        The encoding table is precomputed once and reused for all sequences.
        """
        super().__init__()
        self.model_dimension = model_dimension
        self.dropout = nn.Dropout(dropout)

        # Create matrix to store positional encodings
        # Shape: (max_seq_len, model_dimension)
        positional_encodings_table = torch.zeros(max_seq_len, model_dimension)

        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1) - unsqueeze for broadcasting
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Create dimension indices and compute the divisor term
        # This creates different frequencies for different dimensions
        # Formula: 10000^(-2i/d_model) = exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2).float() * 
            (-math.log(10000.0) / model_dimension)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        positional_encodings_table[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        positional_encodings_table[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_seq_len, model_dimension) → (1, max_seq_len, model_dimension)
        # This allows broadcasting across batch dimension during forward pass
        positional_encodings_table = positional_encodings_table.unsqueeze(0)

        # Register as buffer (not a trainable parameter, but part of module state)
        # This ensures it moves to the correct device (CPU/GPU) with the model
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, x):
        """
        Add positional encodings to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, model_dimension)
        
        Returns:
            torch.Tensor: Embeddings with positional information added
                         Shape: (batch_size, seq_len, model_dimension)
        
        Process:
            1. Extract positional encodings for the sequence length
            2. Add encodings to embeddings (broadcasting handles batch dimension)
            3. Apply dropout for regularization
        """
        # Extract encodings for current sequence length
        # positional_encodings_table is (1, max_seq_len, model_dimension)
        # We slice to (1, seq_len, model_dimension) then broadcast across batch
        positional_encodings = self.positional_encodings_table[:, :x.shape[1], :]
        
        # Add positional information to embeddings
        # Broadcasting: (batch_size, seq_len, d_model) + (1, seq_len, d_model)
        # Ensure positional encodings don't require gradients
        x = x + positional_encodings.requires_grad_(False)
        
        # Apply dropout for regularization
        return self.dropout(x)