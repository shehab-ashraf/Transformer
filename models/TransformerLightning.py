import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from evaluate import load

from models.utils import create_src_mask, create_tgt_mask
from inference.greedy_decode import greedy_decode
from Config import Config


class TransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Transformer training and evaluation.
    
    Handles:
    - Training with teacher forcing
    - Validation with BLEU scoring
    - Learning rate warmup and scheduling
    - Auto logging and checkpointing
    """
    
    def __init__(self, config: Config, transformer: nn.Module, tokenizer):
        """
        Args:
            config: All hyperparameters and settings
            transformer: The Transformer model
            tokenizer: Tokenizer for encoding/decoding text
        """
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.tokenizer = tokenizer

        # Save config for checkpoint resume (exclude large objects)
        self.save_hyperparameters(ignore=['transformer', 'tokenizer'])

        # Loss function: ignores padding, smooths labels
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing, 
            ignore_index=tokenizer.token_to_id("[PAD]")
        )

        # BLEU metric for translation quality
        self.bleu_metric = load("sacrebleu")


    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Run Transformer forward pass.
        
        Args:
            src: Source tokens [batch_size, src_len]
            tgt: Target tokens [batch_size, tgt_len]
            
        Returns:
            Logits [batch_size, tgt_len, vocab_size]
        """
        # Create attention masks
        src_mask = create_src_mask(src).to(self.device)
        tgt_mask = create_tgt_mask(tgt).to(self.device)

        # Encode → Decode → Project to vocabulary
        enc_output = self.transformer.encode(src, src_mask)
        dec_output = self.transformer.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.transformer.project(dec_output)
    

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Single training step with teacher forcing.
        
        Teacher forcing: Feed ground truth tokens as input, predict next token.
        """
        src, tgt = batch
        
        # Shift target: input=[SOS, w1, w2], output=[w1, w2, EOS]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass and loss
        logits = self(src, tgt_input)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),  # [batch*seq, vocab]
            tgt_output.reshape(-1)                 # [batch*seq]
        )

        # Log metrics
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss_step', loss, on_step=True, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, prog_bar=True)

        return loss
    

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step: compute loss and generate translations for BLEU.
        """
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Calculate loss with teacher forcing
        logits = self(src, tgt_input)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), 
            tgt_output.reshape(-1)
        )

        # Generate translations using greedy decoding
        pred = greedy_decode(
            model=self.transformer,
            src=src,
            sos_idx=self.tokenizer.token_to_id("[SOS]"),
            eos_idx=self.tokenizer.token_to_id("[EOS]"),
            device=self.device
        )
        
        # Decode predictions and references to text
        pred_texts = []
        true_texts = []
        pad_idx = self.tokenizer.token_to_id("[PAD]")
        
        for i in range(pred.size(0)):
            # Remove padding tokens
            pred_mask = (pred[i] != pad_idx)
            ref_mask = (tgt_output[i] != pad_idx)
            
            # Convert token IDs to text
            pred_text = self.tokenizer.decode(
                pred[i][pred_mask].tolist(), 
                skip_special_tokens=True
            )
            ref_text = self.tokenizer.decode(
                tgt_output[i][ref_mask].tolist(), 
                skip_special_tokens=True
            )
            
            pred_texts.append(pred_text)
            true_texts.append(ref_text)
    
        # Accumulate for BLEU calculation at epoch end
        if not hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_trues = []
            
        self.all_preds.extend(pred_texts)
        self.all_trues.extend(true_texts)
            
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss


    def on_validation_epoch_end(self):
        """Calculate BLEU score across all validation batches."""
        if hasattr(self, 'all_preds') and self.all_preds:
            # Format: each reference wrapped in list for sacrebleu
            references = [[ref] for ref in self.all_trues]
            
            # Compute BLEU score
            bleu_result = self.bleu_metric.compute(
                predictions=self.all_preds, 
                references=references,
                smooth_method="floor",
                smooth_value=0
            )
            
            self.log('val_bleu', bleu_result['score'], prog_bar=True)
            
            # Clear for next epoch
            self.all_preds = []
            self.all_trues = []


    def configure_optimizers(self) -> dict:
        """
        Setup optimizer and learning rate scheduler.
        Uses Transformer paper's schedule: warmup then decay.
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps
        )

        def lr_lambda(step: int) -> float:
            """
            Learning rate schedule from "Attention Is All You Need".
            
            - Warmup: Linear increase for first N steps
            - Decay: Inverse square root decay after warmup
            
            Formula: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
            """
            d_model = self.config.model.d_model
            warmup_steps = self.config.training.warmup_steps
            step = max(1, step)  # Prevent division by zero
            
            # Scale factor based on model size
            scale = d_model ** (-0.5)
            
            # Warmup: linear growth | Decay: inverse sqrt
            warmup = step * (warmup_steps ** (-1.5))
            decay = step ** (-0.5)
            
            return scale * min(decay, warmup)

        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'  # Update every training step
            }
        }