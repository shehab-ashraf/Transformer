import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.text import BLEUScore
import pytorch_lightning as pl


from Config import Config  # Import Config class


class LabelSmoothingDistribution(nn.Module):
    """
    Custom label smoothing implementation that creates smooth target distributions
    instead of one-hot vectors. This is crucial for transformer training.
    
    Instead of setting target probability to 1.0 for correct token and 0.0 for others,
    we set it to 'confidence_value' (e.g., 0.9) and distribute the remaining 
    'smoothing_value' (e.g., 0.1) evenly across all other tokens.
    
    This prevents the model from becoming overconfident and improves generalization.
    """
    
    def __init__(self, smoothing_value: float, pad_token_id: int, trg_vocab_size: int, device: torch.device):
        """
        Args:
            smoothing_value: How much probability mass to redistribute (e.g., 0.1)
            pad_token_id: Token ID for padding (should not be smoothed)
            trg_vocab_size: Size of target vocabulary
            device: Device to create tensors on
        """
        super(LabelSmoothingDistribution, self).__init__()
        
        # Validate smoothing value
        assert 0.0 <= smoothing_value <= 1.0, f"Smoothing value must be between 0 and 1, got {smoothing_value}"
        
        # Store configuration
        self.confidence_value = 1.0 - smoothing_value  # e.g., 0.9 if smoothing_value=0.1
        self.smoothing_value = smoothing_value          # e.g., 0.1
        self.pad_token_id = pad_token_id
        self.trg_vocab_size = trg_vocab_size
        self.device = device
        
        print(f"LabelSmoothingDistribution initialized:")
        print(f"  - Confidence value: {self.confidence_value}")
        print(f"  - Smoothing value: {self.smoothing_value}")
        print(f"  - Pad token ID: {self.pad_token_id}")
        print(f"  - Vocab size: {self.trg_vocab_size}")

    def forward(self, trg_token_ids_batch: torch.Tensor) -> torch.Tensor:
        """
        Creates smooth target distributions for the given token IDs.
        
        Args:
            trg_token_ids_batch: Shape (batch_size, 1) - ground truth token IDs
            
        Returns:
            smooth_target_distributions: Shape (batch_size, vocab_size) - probability distributions
        """
        batch_size = trg_token_ids_batch.shape[0]
        
        # Initialize all distributions with smoothing value
        # We distribute smoothing mass over (vocab_size - 2) because we exclude:
        # 1. The ground truth token (will be set to confidence_value)
        # 2. The pad token (will be set to 0)
        smooth_target_distributions = torch.zeros((batch_size, self.trg_vocab_size), device=self.device)
        smooth_target_distributions.fill_(self.smoothing_value / (self.trg_vocab_size - 2))
        
        # Set the ground truth token to confidence value
        smooth_target_distributions.scatter_(1, trg_token_ids_batch, self.confidence_value)
        
        # Set pad token probability to 0 (we don't want model to predict padding)
        smooth_target_distributions[:, self.pad_token_id] = 0.
        
        # Special case: if the target token IS the pad token, set entire distribution to 0
        # This prevents the model from learning to predict padding
        smooth_target_distributions.masked_fill_(trg_token_ids_batch == self.pad_token_id, 0.)
        
        return smooth_target_distributions

class TransformerLightning(pl.LightningModule):
    def __init__(
        self,
        config: "Config",
        transformer: nn.Module,
        tokenizer,  # tokenizer object that has decode method
    ):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.tokenizer = tokenizer
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['transformer', 'tokenizer'])
        
        # Initialize loss function - KLDivLoss with custom label smoothing
        # KLDivLoss expects LOG probabilities as input and regular probabilities as target
        # This is why we need log_softmax in the model output
        self.criterion = nn.KLDivLoss(reduction='batchmean')  # 'batchmean' gives better BLEU than 'mean'
        
        # Initialize custom label smoothing distribution
        # Get pad token ID from tokenizer
        pad_token_id = tokenizer.token_to_id("[PAD]")  # Get actual pad token ID
        self.label_smoothing = LabelSmoothingDistribution(
            smoothing_value=config.training.label_smoothing,
            pad_token_id=pad_token_id,
            trg_vocab_size=tokenizer.get_vocab_size(),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        print(f"Loss function initialized:")
        print(f"  - Using KLDivLoss with reduction='batchmean'")
        print(f"  - Label smoothing value: {config.training.label_smoothing}")
        print(f"  - Pad token ID: {pad_token_id}")
        print(f"  - Target vocab size: {tokenizer.get_vocab_size()}")
        
        # Initialize BLEU metric
        self.bleu = BLEUScore(n_gram=4)
        
    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        
        enc_output = self.transformer.encode(src, src_mask)
        dec_output = self.transformer.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.transformer.project(dec_output)
    
    def create_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def create_tgt_mask(self, tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask.to(self.device)
        nopeak_mask = nopeak_mask.to(self.device)
        return tgt_mask & nopeak_mask

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]  # Input to decoder (without last token)
        tgt_output = tgt[:, 1:]   # Target output (without first token)
        
        # Forward pass - model now outputs LOG probabilities
        log_probs = self(src, tgt_input)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        # log_probs: (batch_size * seq_len, vocab_size)
        # tgt_output: (batch_size * seq_len, 1) - we need to reshape to (batch_size * seq_len, 1)
        log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        tgt_output_flat = tgt_output.reshape(-1, 1)  # Keep as (N, 1) for label smoothing
        
        # Create smooth target distributions
        smooth_targets = self.label_smoothing(tgt_output_flat)  # Shape: (batch_size * seq_len, vocab_size)
        
        # Compute KL divergence loss
        # KLDivLoss expects: input=log_probs, target=probs
        loss = self.criterion(log_probs_flat, smooth_targets)
        
        # Get current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        # Step-level logging with comprehensive metrics
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]  # Input to decoder (without last token)
        tgt_output = tgt[:, 1:]   # Target output (without first token)
        
        # Forward pass - model now outputs LOG probabilities
        log_probs = self(src, tgt_input)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        tgt_output_flat = tgt_output.reshape(-1, 1)  # Keep as (N, 1) for label smoothing
        
        # Create smooth target distributions
        smooth_targets = self.label_smoothing(tgt_output_flat)  # Shape: (batch_size * seq_len, vocab_size)
        
        # Compute KL divergence loss
        loss = self.criterion(log_probs_flat, smooth_targets)
        
        # Get predictions (convert log_probs back to regular probabilities for argmax)
        # We need to convert log_probs to regular probs for argmax
        probs = torch.exp(log_probs)  # Convert log_probs to probs
        pred = probs.argmax(-1)  # Get predicted token IDs
        
        # Decode predictions and targets using tokenizer
        for pred_seq, ref_seq in zip(pred, tgt_output):
            # Remove padding tokens and decode
            pred_mask = pred_seq != 0
            ref_mask = ref_seq != 0
            
            # Decode using tokenizer
            pred_text = self.tokenizer.decode(pred_seq[pred_mask].tolist())
            ref_text = self.tokenizer.decode(ref_seq[ref_mask].tolist())
            
            # Update BLEU metric state
            self.bleu.update([pred_text], [[ref_text]])
            
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log the final BLEU score
        bleu_score = self.bleu.compute()
        self.log('val_bleu_epoch', bleu_score, prog_bar=True)
        # Reset metric for next epoch
        self.bleu.reset()

    def test_step(self, batch, batch_idx):
        """Test step for final evaluation on test set"""
        src, tgt = batch
        tgt_input = tgt[:, :-1]  # Input to decoder (without last token)
        tgt_output = tgt[:, 1:]   # Target output (without first token)
        
        # Forward pass - model now outputs LOG probabilities
        log_probs = self(src, tgt_input)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        tgt_output_flat = tgt_output.reshape(-1, 1)  # Keep as (N, 1) for label smoothing
        
        # Create smooth target distributions
        smooth_targets = self.label_smoothing(tgt_output_flat)  # Shape: (batch_size * seq_len, vocab_size)
        
        # Compute KL divergence loss
        loss = self.criterion(log_probs_flat, smooth_targets)
        
        # Get predictions (convert log_probs back to regular probabilities for argmax)
        probs = torch.exp(log_probs)  # Convert log_probs to probs
        pred = probs.argmax(-1)  # Get predicted token IDs
        
        # Decode predictions and targets using tokenizer
        for pred_seq, ref_seq in zip(pred, tgt_output):
            # Remove padding tokens and decode
            pred_mask = pred_seq != 0
            ref_mask = ref_seq != 0
            
            # Decode using tokenizer
            pred_text = self.tokenizer.decode(pred_seq[pred_mask].tolist())
            ref_text = self.tokenizer.decode(ref_seq[ref_mask].tolist())
            
            # Update BLEU metric state
            self.bleu.update([pred_text], [[ref_text]])
            
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        """Compute and log final test BLEU score"""
        # Compute and log the final test BLEU score
        test_bleu_score = self.bleu.compute()
        self.log('test_bleu_epoch', test_bleu_score, prog_bar=True)
        
        # Reset metric for next run
        self.bleu.reset()

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps,
        )
        
        def lr_lambda(step):
            # Implementation of formula from Section 5.3 of Transformer paper
            # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            d_model = self.config.model.d_model
            warmup_steps = self.config.training.warmup_steps
            step = max(1, step)  # Avoid division by zero
            factor = d_model ** (-0.5)
            arg1 = step ** (-0.5)
            arg2 = step * warmup_steps ** (-1.5)
            return factor * min(arg1, arg2)
            
        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }