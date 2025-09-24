import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.text import BLEUScore
import pytorch_lightning as pl


from Config import Config  # Import Config class

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
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing, 
            ignore_index=0  # assuming 0 is pad_idx
        )
        
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
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = self(src, tgt_input)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Get current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        # Step-level logging with comprehensive metrics
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = self(src, tgt_input)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Get predictions
        pred = logits.argmax(-1)
        
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

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps
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