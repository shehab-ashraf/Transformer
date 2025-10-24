"""
This module implements the embedding layer that converts token indices into continuous dense vector representations.

Key Concept:
------------
- Tokens (words/subwords) are represented as integers (indices in vocabulary)
- This layer maps each index to a learned dense vector of size `model_dimension`
- Embeddings are scaled by sqrt(model_dimension)
"""

import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Converts discrete token indices into dense continuous vector representations.
    
    This is the first layer in the Transformer that processes input sequences.
    It learns a lookup table where each vocabulary token has a corresponding
    dense vector representation.
    
    Attributes:
        model_dimension (int): Size of the embedding vectors (d_model)
        embedding (nn.Embedding): PyTorch embedding layer mapping vocab → vectors
    """

    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        """
        Initialize the embedding layer.
        
        Args:
            model_dimension (int): Dimensionality of embedding vectors 
            vocab_size (int): Total number of unique tokens in vocabulary
        
        Note:
            The embedding weights are randomly initialized and learned during training.
            Weight shape: (vocab_size, model_dimension)
        """
        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        
        # Create embedding lookup table: vocab_size × model_dimension
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):
        """
        Convert token indices to scaled embeddings.
        
        Args:
            x (torch.Tensor): Token indices of shape (batch_size, seq_len)
                             Values must be in range [0, vocab_size-1]
        
        Returns:
            torch.Tensor: Embedded representations of shape (batch_size, seq_len, model_dimension)
                         Each token index is replaced with its learned embedding vector,
                         scaled by sqrt(model_dimension)
        """
        # Lookup embeddings and scale by sqrt(d_model)
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, model_dimension)
        return self.embedding(x) * math.sqrt(self.model_dimension)