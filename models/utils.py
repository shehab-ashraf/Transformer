    
import torch

def create_src_mask(src: torch.Tensor) -> torch.Tensor:
    """
    Create source attention mask to ignore padding tokens.
        
    Args:
        src: Source tensor of shape [batch_size, src_seq_len]
    
    Returns:
        Boolean mask of shape [batch_size, 1, 1, src_seq_len]
    """
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    return src_mask

def create_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
    """
    Create target attention mask with no-peak (causal) and padding masking.
        
    Args:
        tgt: Target tensor of shape [batch_size, tgt_seq_len]
            
    Returns:
        Boolean mask of shape [batch_size, 1, tgt_seq_len, tgt_seq_len]
    """
    # Padding mask
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    # Causal mask to prevent looking ahead
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        
    return tgt_mask & nopeak_mask