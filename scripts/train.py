import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import os

from models.TransformerBlocks.Transformer import Transformer
from models.TransformerLightning import TransformerLightning
from data.iwslt2017 import IWSLT2017DataModule
from Config import Config


def train():

    print("Starting Transformer training with comprehensive logging")
    
    # Initialize configuration
    config = Config()
    
    # Key settings for good performance on IWSLT2017
    config.model.d_model = 512
    config.model.n_heads = 8
    config.model.n_layers = 6
    config.model.d_ff = 2048
    config.model.dropout = 0.3  # Increased dropout for better regularization
    
    config.data.batch_size = 32  
    config.data.max_seq_len = 128  
    config.data.data_size = 200000
    config.data.test_proportion = 0.1
    
    config.training.max_epochs = 30  
    config.training.warmup_steps = 8000  # More warmup steps for stability
    config.training.label_smoothing = 0.1
    config.training.accumulate_grad_batches = 4  

    # Log configuration
    print(f"Training configuration: {config}")
    print(f"Model config: d_model={config.model.d_model}, n_heads={config.model.n_heads}, n_layers={config.model.n_layers}")
    print(f"Data config: batch_size={config.data.batch_size}, max_seq_len={config.data.max_seq_len}")
    print(f"Training config: max_epochs={config.training.max_epochs}, warmup_steps={config.training.warmup_steps}")

    # Initialize data module
    print("Initializing data module...")

    datamodule = IWSLT2017DataModule(
        batch_size=config.data.batch_size,
        num_workers=4,
        pin_memory=True,
        data_size=config.data.data_size,
        max_seq_len=config.data.max_seq_len,
        test_proportion=config.data.test_proportion,
        vocab_size=config.tokenizer.vocab_size
    )
    
    # Prepare the data (tokenizer and preprocessing)
    print("Preparing data and tokenizer...")
    datamodule.prepare()
    print(f"Data preparation complete. Vocab size: {datamodule.tokenizer.vocab_size}")
    
    # Create transformer model
    print("Creating Transformer model...")
    transformer = Transformer(
        src_vocab_size=config.tokenizer.vocab_size,
        tgt_vocab_size=config.tokenizer.vocab_size,
        src_seq_len=config.data.max_seq_len,
        tgt_seq_len=config.data.max_seq_len,
        model_dimension=config.model.d_model,
        N=config.model.n_layers,
        h=config.model.n_heads,
        dropout=config.model.dropout,
        d_ff=config.model.d_ff
    )
    
    # Count model parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    # Create lightning module
    print("Creating Lightning module...")
    model = TransformerLightning(
        config=config,
        transformer=transformer,
        tokenizer=datamodule.tokenizer
    )
    
    # Setup wandb logging
    print("Setting up Weights & Biases logging...")
    wandb_logger = WandbLogger(
        project="transformer",
        name="iwslt2017",
        log_model=True,
        save_dir="logs"
    )
    
    # Callbacks
    callbacks = [
        # Stop if not improving
        EarlyStopping(
            patience=5,
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.01
        )
    ]
    
    # Initialize trainer
    print("Initializing PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,  # Log every 10 steps
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        precision=16,  # Use mixed precision for faster training
    )
    
    # Log training setup
    print(f"Trainer configured with {trainer.num_devices} device(s)")
    print(f"Training will run for {config.training.max_epochs} epochs")
    print(f"Gradient accumulation: {config.training.accumulate_grad_batches} batches")
    
    # Train model
    print("Starting training...")

    trainer.fit(model, datamodule=datamodule)
    
    # Close wandb
    wandb.finish()
    print("Training completed successfully!")

if __name__ == "__main__":
    train()