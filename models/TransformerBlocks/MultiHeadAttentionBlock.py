import torch.nn as nn
import math

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
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dimension = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)

        if mask is not None: 
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.number_of_heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.number_of_heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.number_of_heads, self.head_dimension).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.number_of_heads * self.head_dimension)

        return self.w_o(x)