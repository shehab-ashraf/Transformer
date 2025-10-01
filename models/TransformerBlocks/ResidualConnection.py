import torch.nn as nn


class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization and dropout.
    
    Implements: x + Dropout(Sublayer(LayerNorm(x)))
    
    Note: Uses Pre-LN (normalize before sublayer), which is more stable
    than the original Post-LN from the paper.
    """

    def __init__(self, features: int, dropout: float) -> None:
        """
        Args:
            features: Model dimension (d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)
    

    def forward(self, x, sublayer):
        """
        Apply residual connection around a sublayer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            sublayer: Function to apply (attention or feed-forward)
            
        Returns:
            Output with residual [batch_size, seq_len, d_model]
        """
        # Normalize → Apply sublayer → Dropout → Add to original input
        return x + self.dropout(sublayer(self.norm(x)))