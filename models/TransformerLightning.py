"""
Training and evaluating Transformer model using PyTorch Lightning framework. 
This module encapsulates the complete training pipeline, implementing 
best practices from "Attention Is All You Need" (Vaswani et al., 2017).

What This Module Does:
----------------------
• Teacher Forcing: Uses correct tokens as input during training
• Greedy Decoding: Generates actual translations during validation
• Learning Rate: Warmup then decay (from the original Transformer paper)
• BLEU Scoring: Measures translation quality 
• Label Smoothing: Makes the model less overconfident

During validation, we do TWO things:
1. Calculate loss with teacher forcing → How well model predicts
2. Generate translations with greedy decoding → How good translations are

Why both? Because low loss doesn't always mean good translations!
The model might memorize patterns without learning to translate well.

Learning Rate Schedule:
-----------------------
Uses the schedule from "Attention Is All You Need":
    lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

• Warmup: Slowly increase learning rate for first ~4000 steps
  Why? Prevents big unstable gradients at the start
  
• Decay: Gradually decrease after warmup (inverse square root)
  Why? Learn fast at first, then fine-tune slowly
"""

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

    def __init__(self, config: Config, transformer: nn.Module, tokenizer):
        """
        Initialize the Lightning module.

        Parameters
        ----------
        config : Config
            Configuration object containing all model and training hyperparameters.
        transformer : nn.Module
            The Transformer model instance to train and evaluate.
        tokenizer : object
            Tokenizer object for converting between tokens and text.
        """
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.tokenizer = tokenizer

        # Save configuration for checkpoint compatibility
        self.save_hyperparameters(ignore=['transformer', 'tokenizer'])

        # Define loss: cross-entropy with label smoothing and PAD ignored
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing,
            ignore_index=tokenizer.token_to_id("[PAD]")
        )

        # Initialize BLEU metric for evaluation
        self.bleu_metric = load("sacrebleu")



    ##----------------------Forward Pass------------------------------##
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Parameters
        ----------
        src : torch.Tensor
            Source token IDs, shape [batch_size, src_len].
        tgt : torch.Tensor
            Target token IDs, shape [batch_size, tgt_len].

        Returns
        -------
        torch.Tensor
            Logits over vocabulary, shape [batch_size, tgt_len, vocab_size].
        """
        # Create attention masks
        src_mask = create_src_mask(src).to(self.device)
        tgt_mask = create_tgt_mask(tgt).to(self.device)

        # Encode → Decode → Project
        enc_output = self.transformer.encode(src, src_mask)
        dec_output = self.transformer.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.transformer.project(dec_output)



    ##----------------------Training Step------------------------------##
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step using **teacher forcing**.

        Teacher forcing means using the ground truth sequence as the decoder input
        and training the model to predict the next token at each step.

        Parameters
        ----------
        batch : tuple
            A tuple (src, tgt) of tokenized input and target sequences.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed training loss for the current batch.
        """
        src, tgt = batch

        # Shift target sequence for input/output pairs
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = self(src, tgt_input)

        # Compute loss
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),  # [batch * seq_len, vocab_size]
            tgt_output.reshape(-1)                # [batch * seq_len]
        )

        # Log metrics
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, prog_bar=True)

        return loss


    ##----------------------Validation Step------------------------------##
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Computes validation loss and generates predictions using greedy decoding.
        Predictions and references are collected for BLEU computation.
        
        Parameters
        ----------
        batch : tuple
            A tuple (src, tgt) of tokenized source and target sequences.
        batch_idx : int
            Index of the current validation batch.

        Returns
        -------
        torch.Tensor
            Validation loss for the current batch.
        """
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Compute loss
        logits = self(src, tgt_input)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        # Generate predictions
        pred = greedy_decode(
            model=self.transformer,
            src=src,
            max_length=50,
            sos_idx=self.tokenizer.token_to_id("[SOS]"),
            eos_idx=self.tokenizer.token_to_id("[EOS]"),
            device=self.device
        )

        # Decode predictions and references
        pred_texts, true_texts = [], []
        pad_idx = self.tokenizer.token_to_id("[PAD]")

        for i in range(pred.size(0)):
            pred_mask = (pred[i] != pad_idx)
            ref_mask = (tgt_output[i] != pad_idx)

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

        # Accumulate predictions for BLEU computation
        if not hasattr(self, 'all_preds'):
            self.all_preds, self.all_trues = [], []

        self.all_preds.extend(pred_texts)
        self.all_trues.extend(true_texts)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss


    ##----------------------Validation Epoch End------------------------------##
    def on_validation_epoch_end(self):
        """
        Compute the BLEU score over the entire validation set.
        This is called once per validation epoch, after all batches are processed.
        """
        if hasattr(self, 'all_preds') and self.all_preds:
            # BLEU expects references as list of lists
            references = [[ref] for ref in self.all_trues]

            bleu_result = self.bleu_metric.compute(
                predictions=self.all_preds,
                references=references,
                smooth_method="floor",
                smooth_value=0
            )

            self.log('val_bleu', bleu_result['score'], prog_bar=True)

            # Reset for next epoch
            self.all_preds, self.all_trues = [], []


    ##----------------------Optimizer and Scheduler------------------------------##
    def configure_optimizers(self) -> dict:
        """
        Configure optimizer and learning rate scheduler.

        Uses the same learning rate schedule described in
        **"Attention Is All You Need" (Vaswani et al., 2017)**:

        This schedule:
        - Linearly increases the learning rate during the warmup phase.
        - Then decays proportionally to the inverse square root of the step number.

        Returns
        -------
        dict
            Dictionary containing optimizer and LR scheduler for PyTorch Lightning.
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps
        )

        def lr_lambda(step: int) -> float:
            """Lambda function implementing the Transformer learning rate schedule."""
            d_model = self.config.model.d_model
            warmup_steps = self.config.training.warmup_steps
            step = max(1, step)  # avoid division by zero

            scale = d_model ** (-0.5)
            warmup = step * (warmup_steps ** (-1.5))
            decay = step ** (-0.5)

            return scale * min(decay, warmup)

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'  # Update LR every step
            }
        }
