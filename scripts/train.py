"""
Main training script for Transformer on IWSLT2017 dataset.

Handles:
- Argument parsing.
- Config creation and overriding.
- Data, model, and trainer setup.
- Logging, checkpointing, and early stopping.
"""
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.TransformerBlocks.Transformer import Transformer
from models.TransformerLightning import TransformerLightning
from data.iwslt2017 import IWSLT2017DataModule
from Config import Config

##===================Parse CLI Arguments=====================##
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer on IWSLT2017")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation (effective batch size multiplier)')
    parser.add_argument('--learning_rate', type=float, default=1, help='Learning rate')
    
    # Utility args
    parser.add_argument('--run_name', type=str, default='iwslt2017', help='Experiment name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


##=====================Train Function=======================##
def train(args):
    """Train the model."""
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # ---------------- CONFIG ----------------
    config = Config()
    config.data.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.learning_rate = args.learning_rate
    config.training.accumulate_grad_batches = args.accumulate_grad_batches
    
    # ---------------- DATA ----------------
    print("Loading data...")
    datamodule = IWSLT2017DataModule(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        max_seq_len=config.data.max_seq_len,
        vocab_size=config.tokenizer.vocab_size
    )
    datamodule.prepare()
    
    # ---------------- MODEL ----------------
    print("Building model...")
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
    
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total Parameters: {total_params:,}")

    model = TransformerLightning(config=config, transformer=transformer, tokenizer=datamodule.tokenizer)

    # ---------------- LOGGING ----------------
    logger = None if args.no_wandb else WandbLogger(project="Transformer", name=args.run_name, log_model=True)

    # ---------------- CALLBACKS ----------------
    callbacks = [
        ModelCheckpoint(
            dirpath=config.paths.checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.2f}",
            monitor="val_bleu",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_bleu",
            patience=5,
            mode="max",
            min_delta=0.01
        ),
    ]
    
    # ---------------- TRAINER ----------------
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=10
    )

    print(f"Training for {args.max_epochs} epochs "f"with batch size {args.batch_size} (x{args.accumulate_grad_batches})")
    
    trainer.fit(model, datamodule=datamodule)

    print("Training complete!") 
    if not args.no_wandb: wandb.finish()


##================Entry Point===================##
if __name__ == "__main__":
    args = parse_args()
    train(args)