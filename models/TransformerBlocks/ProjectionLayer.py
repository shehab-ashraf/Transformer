import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    Final projection layer of the Transformer.
    
    Purpose:
    - Converts decoder outputs (hidden representations) into vocabulary scores.
    - Each hidden vector is mapped to a vector the size of the vocabulary.
    """

    def __init__(self, model_dimension, vocab_size) -> None: 
        super().__init__()
        # Linear layer projects from hidden dimension → vocab size
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Decoder representations 
            Shape → (batch_size, seq_len, model_dimension)
            
        Returns:
            logits: Vocabulary scores
            Shape → (batch_size, seq_len, vocab_size)
        """
        # Apply linear projection
        return self.proj(x)
