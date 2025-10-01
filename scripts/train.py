import argparse
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.TransformerBlocks.Transformer import Transformer
from models.TransformerLightning import TransformerLightning
from data.iwslt2017 import IWSLT2017DataModule
from Config import Config


def parse_args():
    """Parse command line arguments - only the ones you actually change."""
    parser = argparse.ArgumentParser(description="Train Transformer on IWSLT2017")
    
    # Things you actually tune during experiments
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, 
                       help='Gradient accumulation (effective batch size multiplier)')
    parser.add_argument('--max_seq_len', type=int, default=100, help='Max sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # Utility args
    parser.add_argument('--run_name', type=str, default='iwslt2017', help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/models', help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def train(args):
    """Train the model."""
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    config = Config()
    
    # Override config with CLI args (only what's needed)
    config.data.batch_size = args.batch_size
    config.data.max_seq_len = args.max_seq_len
    config.training.max_epochs = args.max_epochs
    config.training.learning_rate = args.learning_rate
    config.training.accumulate_grad_batches = args.accumulate_grad_batches
    
    # ========================================================================
    # DATA
    # ========================================================================
    print("Loading data...")
    datamodule = IWSLT2017DataModule(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        max_seq_len=config.data.max_seq_len,
        vocab_size=config.tokenizer.vocab_size
    )
    datamodule.prepare()
    
    # ========================================================================
    # MODEL
    # ========================================================================
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
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Wrap in Lightning
    model = TransformerLightning(
        config=config,
        transformer=transformer,
        tokenizer=datamodule.tokenizer
    )
    
    # ========================================================================
    # LOGGER & CALLBACKS
    # ========================================================================
    logger = None if args.no_wandb else WandbLogger(
        project="Transformer",
        name=args.run_name,
        log_model=True
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
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
    
    # ========================================================================
    # TRAINER
    # ========================================================================
    print(f"Training for {args.max_epochs} epochs...")
    print(f"Batch size: {args.batch_size} x {args.accumulate_grad_batches} = {args.batch_size * args.accumulate_grad_batches}")
    
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
    
    # ========================================================================
    # TRAIN
    # ========================================================================
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    
    print("Training complete!")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)