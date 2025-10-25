from dataclasses import dataclass, field
from typing import Tuple


##==========================Tokenizer Configuration=================================##
@dataclass
class TokenizerConfig:
    """
    Configuration for tokenizer training and vocabulary.

    Attributes:
        vocab_size (int): vocabulary size after BPE training.
        min_frequency (int): Minimum frequency required to include a token.
    """
    vocab_size: int = 24000
    min_frequency: int = 3


##==========================Data Configuration=================================##
@dataclass
class DataConfig:
    """
    Configuration for dataset loading and batching.

    Attributes:
        batch_size (int): Number of samples per training batch.
        num_workers (int): Number of CPU workers for DataLoader.
        pin_memory (bool): Enables faster GPU transfer if True.
        max_seq_len (int): Maximum tokenized sequence length.
    """
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True
    max_seq_len: int = 100


##==========================Model Configuration=================================##
@dataclass
class ModelConfig:
    """
    Configuration for Transformer model architecture.

    Attributes:
        d_model (int): Hidden dimension (embedding size).
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder/decoder layers.
        d_ff (int): Dimension of feedforward layer.
        dropout (float): Dropout probability.
    """
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1


##==========================Training Configuration=================================##
@dataclass
class TrainingConfig:
    """
    Configuration for training process and optimization.

    Attributes:
        max_epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        warmup_steps (int): Number of LR warmup steps.
        gradient_clip_val (float): Max gradient norm.
        label_smoothing (float): Label smoothing factor.
        accumulate_grad_batches (int): Gradient accumulation steps.
        optimizer_betas (Tuple[float, float]): Adam betas.
        optimizer_eps (float): Adam epsilon.
    """
    max_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.1
    accumulate_grad_batches: int = 1
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-9

##==========================Paths Configuration=================================##
@dataclass
class PathsConfig:
    """
    Configuration for directory paths

    Attributes:
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    checkpoint_dir: str = "checkpoints/model"



##==========================configuration container=================================##
@dataclass
class Config:
    """
    configuration container holding all sub-configs.
    """
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
