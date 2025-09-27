import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from models.TransformerBlocks.Transformer import Transformer
from models.TransformerLightning import TransformerLightning
from data.iwslt2017 import IWSLT2017DataModule
from Config import Config

def train():
    # Initialize configuration
    config = Config()
    
    # Key settings for good performance on IWSLT2017
    config.model.d_model = 512
    config.model.n_heads = 8
    config.model.n_layers = 6
    config.model.d_ff = 2048
    config.model.dropout = 0.1
    
    config.data.batch_size = 32  
    config.data.max_seq_len = 100  
    
    config.training.max_epochs = 20 
    config.training.warmup_steps = 4000  
    config.training.label_smoothing = 0.1
    config.training.accumulate_grad_batches = 2

    # Initialize data module
    datamodule = IWSLT2017DataModule(
        batch_size=config.data.batch_size,
        num_workers=4,
        pin_memory=True,
        max_seq_len=config.data.max_seq_len,
        vocab_size=config.tokenizer.vocab_size
    )
    
    # Prepare the data (tokenizer and preprocessing)
    datamodule.prepare()
    
    # Create transformer model
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
    
    # Create lightning module
    model = TransformerLightning(
        config=config,
        transformer=transformer,
        tokenizer=datamodule.tokenizer
    )
    
    # Setup wandb logging
    wandb_logger = WandbLogger(
        project="Transformer",
        name="iwslt2017",
        log_model=True
    )
    
    # Callbacks
    callbacks = [
        # Save best models
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="transformer-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        # Stop if not improving
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.01
        ),
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    
    # Train model
    trainer.fit(model, datamodule=datamodule)
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    train()