"""
This module implements the core attention mechanism that allows the model to focus
on different parts of the input sequence when processing each token.

Key Concepts:
-------------
1. **Attention**: Mechanism to weigh the importance of different tokens
2. **Multi-Head**: Run attention multiple times in parallel with different learned projections
3. **Scaled Dot-Product**: Compute attention scores using query-key similarity

Why Multi-Head?
---------------
- Different heads can learn different types of relationships
- Some heads might focus on syntactic patterns, others on semantic relationships
- Provides richer representations than single attention mechanism

Attention Formula:
------------------
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
    Q (Query): "What am I looking for?"
    K (Key): "What do I offer?"
    V (Value): "What do I actually contain?"
    d_k: Dimension per head (for scaling)

Multi-Head Process:
-------------------
1. Project input into Q, K, V using learned weights
2. Split into h heads (each with dimension d_k = d_model/h)
3. Apply scaled dot-product attention for each head independently
4. Concatenate all head outputs
5. Project back to d_model dimension
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    """
    Attributes:
        model_dimension (int): Total embedding dimension (d_model)
        number_of_heads (int): Number of parallel attention heads (h)
        head_dimension (int): Dimension per head (d_k = d_model / h)
        w_q, w_k, w_v (nn.Linear): Projection matrices for queries, keys, values
        w_o (nn.Linear): Output projection after concatenating heads
        dropout (nn.Dropout): Dropout layer for attention weights
        attention_scores (torch.Tensor): Stored attention weights for visualization
    """

    def __init__(self, model_dimension: int, number_of_heads: int, dropout: float) -> None:
        """
        Initialize Multi-Head Attention block.
        
        Args:
            model_dimension (int): Embedding dimension
            number_of_heads (int): Number of attention heads
            dropout (float): Dropout probability for attention weights
        
        Raises:
            AssertionError: If model_dimension is not divisible by number_of_heads
        
        Note:
            The projection matrices are initialized randomly and learned during training.
            Using bias=False follows the original Transformer paper.
        """
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads

        # Ensure model dimension divides evenly across heads
        assert model_dimension % number_of_heads == 0, \
            f'Model dimension ({model_dimension}) must be divisible by number of heads ({number_of_heads})'

        # Calculate dimension per head
        self.head_dimension = model_dimension // number_of_heads

        # Linear projection layers for queries, keys, and values
        # Each projects from d_model to d_model (then split into h heads)
        self.w_q = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_k = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_v = nn.Linear(model_dimension, model_dimension, bias=False)

        # Output projection after concatenating all heads
        self.w_o = nn.Linear(model_dimension, model_dimension, bias=False)

        # Dropout for regularization on attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Attention scores for visualization
        self.attention_scores = None

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute Scaled Dot-Product Attention.
        
        This is the core attention mechanism that computes weighted combinations
        of values based on the similarity between queries and keys.
        
        Args:
            query (torch.Tensor): Queries of shape (batch, heads, seq_len, d_k)
            key (torch.Tensor): Keys of shape (batch, heads, seq_len, d_k)
            value (torch.Tensor): Values of shape (batch, heads, seq_len, d_k)
            mask (torch.Tensor, optional): Mask of shape (batch, 1, seq_len, seq_len)
            dropout (nn.Dropout, optional): Dropout layer to apply on attention weights
        
        Returns:
            tuple: (output, attention_scores)
                - output: Weighted sum of values (batch, heads, seq_len, d_k)
                - attention_scores: Attention weights (batch, heads, seq_len, seq_len)
        
        Process:
            1. Compute similarity scores: QK^T
            2. Scale by sqrt(d_k)
            3. Apply mask to certain positions (padding, future tokens)
            4. Apply softmax to get probability distribution
            5. Apply dropout for regularization
            6. Compute weighted sum of values
        """
        # Get dimension per head for scaling
        head_dimension = query.shape[-1]

        # Step 1 & 2: Compute scaled attention scores
        # (batch, h, seq, d_k) @ (batch, h, d_k, seq) → (batch, h, seq, seq)
        # Scaling by sqrt(d_k) prevents dot products from growing too large
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)

        # Step 3: Apply mask if provided
        # Set masked positions to large negative value so softmax → 0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Step 4: Convert scores to probabilities via softmax
        # Each query position gets a probability distribution over all key positions
        # Shape remains: (batch, heads, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)

        # Step 5: Apply dropout to attention weights (regularization technique)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Step 6: Compute weighted sum of values
        # (batch, h, seq, seq) @ (batch, h, seq, d_k) → (batch, h, seq, d_k)
        output = attention_scores @ value

        return output, attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Forward pass through Multi-Head Attention.
        
        Args:
            q (torch.Tensor): Query input (batch_size, seq_len, model_dimension)
            k (torch.Tensor): Key input (batch_size, seq_len, model_dimension)
            v (torch.Tensor): Value input (batch_size, seq_len, model_dimension)
            mask (torch.Tensor, optional): Attention mask
        
        Returns:
            torch.Tensor: Output after multi-head attention (batch_size, seq_len_q, model_dimension)
        
        Use Cases:
            - Self-Attention: q=k=v (tokens attend to each other in same sequence)
            - Cross-Attention: q≠k=v (decoder attends to encoder output)
        
        Process:
            1. Project inputs to Q, K, V spaces
            2. Reshape to separate heads: (batch, seq, d_model) → (batch, heads, seq, d_k)
            3. Apply attention for each head in parallel
            4. Concatenate heads back together
            5. Apply final output projection
        """
        # Step 1: Linear projections
        # Each transforms (batch, seq_len, d_model) → (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Step 2: Reshape and split into multiple heads
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        batch_size = query.shape[0]
        query = query.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        key = key.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        value = value.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention for all heads in parallel
        # Each head operates on its own (d_k)-dimensional subspace
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Step 4: Concatenate all heads back together
        # (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k) → (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dimension)

        # Step 5: Apply final linear projection
        # This mixes information from different heads
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        return self.w_o(x)