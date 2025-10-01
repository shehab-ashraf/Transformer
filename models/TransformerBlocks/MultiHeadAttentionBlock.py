import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention block as used in the Transformer.
    
    Core idea:
    - Split input representations into multiple heads
    - Each head performs scaled dot-product attention independently
    - Concatenate results and project back to the model dimension
    
    This allows the model to jointly attend to information 
    from different representation subspaces at different positions.
    """

    def __init__(self, model_dimension: int, number_of_heads: int, dropout: float) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads

        # Ensure model dimension divides evenly across heads
        assert model_dimension % number_of_heads == 0, \
            f'Model dimension must be divisible by the number of heads.'

        # Dimension of each individual head
        self.head_dimension = model_dimension // number_of_heads

        # Learnable linear projections for queries, keys, and values
        self.w_q = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_k = nn.Linear(model_dimension, model_dimension, bias=False)
        self.w_v = nn.Linear(model_dimension, model_dimension, bias=False)

        # Final projection after concatenating all heads
        self.w_o = nn.Linear(model_dimension, model_dimension, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Scaled Dot-Product Attention:
        - Compute attention scores as QK^T / sqrt(d_k)
        - Apply mask (if provided) to prevent attending to certain positions
        - Normalize with softmax
        - Apply dropout
        - Weighted sum over values
        """
        head_dimension = query.shape[-1]

        # Raw attention scores (batch, heads, seq_len_q, seq_len_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)

        # Apply mask: set disallowed positions to very negative values
        if mask is not None: 
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Softmax normalization to get attention distribution
        attention_scores = attention_scores.softmax(dim=-1)

        # Optional dropout on attention weights
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # Weighted sum of values
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Args:
            q, k, v: Input tensors of shape (batch, seq_len, model_dimension)
            mask: Optional mask to block certain positions (e.g., future tokens)

        Returns:
            Output of shape (batch, seq_len, model_dimension)
        """
        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape into (batch, heads, seq_len, head_dim)
        query = query.view(query.shape[0], query.shape[1], self.number_of_heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.number_of_heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.number_of_heads, self.head_dimension).transpose(1, 2)

        # Apply scaled dot-product attention for all heads
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate all heads back together
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.number_of_heads * self.head_dimension)

        # Final linear projection
        return self.w_o(x)
