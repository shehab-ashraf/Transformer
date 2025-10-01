import torch
from models.utils import create_src_mask, create_tgt_mask


def greedy_decode(
    model, 
    src: torch.Tensor, 
    max_length: int = 60, 
    sos_idx: int = 1, 
    eos_idx: int = 2, 
    device="cpu"
):
    """
    Generate translations using greedy decoding (always pick most likely token).

    Args:
        model: Trained Transformer model
        src: Source sequences [batch_size, src_seq_len]
        max_length: Maximum tokens to generate
        sos_idx: Start of sequence token ID
        eos_idx: End of sequence token ID
        device: Device to run inference on

    Returns:
        Generated sequences [batch_size, generated_seq_len]
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    src = src.to(device)

    with torch.no_grad():  # No gradient calculation needed for inference
        # Encode source sequence once (reused for all decoding steps)
        src_mask = create_src_mask(src).to(device)
        enc_output = model.encode(src, src_mask)

        # Initialize target with start token [batch_size, 1]
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        # Generate tokens one at a time
        for _ in range(max_length - 1):
            # Decode current target sequence
            tgt_mask = create_tgt_mask(tgt).to(device)
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
            
            # Project decoder output to vocabulary and get token probabilities
            logits = model.project(dec_output)

            # Pick token with highest probability from last position
            next_token = logits[:, -1:].argmax(-1)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if all sequences generated end token
            if (next_token == eos_idx).all():
                break

        return tgt