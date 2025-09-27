import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    Final projection layer that converts model representations to vocabulary log probabilities.
    
    This layer applies:
    1. Linear projection from model_dimension to vocab_size
    2. Log softmax to get log probabilities (required for KLDivLoss)
    """

    def __init__(self, model_dimension, vocab_size) -> None: 
        super().__init__()
        self.proj = nn.Linear(model_dimension, vocab_size)
        # Log softmax for KLDivLoss compatibility
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: Model representations of shape (batch_size, seq_len, model_dimension)
            
        Returns:
            log_probs: Log probabilities of shape (batch_size, seq_len, vocab_size)
        """
        # Apply linear projection
        logits = self.proj(x)
        # Convert to log probabilities (required for KLDivLoss)
        log_probs = self.log_softmax(logits)
        return log_probs