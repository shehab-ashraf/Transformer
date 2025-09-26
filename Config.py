from dataclasses import dataclass, field
from typing import List

@dataclass
class TokenizerConfig:
    vocab_size: int = 12000
    min_frequency: int = 3
    save_dir: str = "checkpoints/tokenizer"

@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    max_seq_len: int = 128

@dataclass
class ModelConfig:
    d_model: int = 512  # paper default
    n_heads: int = 8    # paper default
    n_layers: int = 6   # paper default
    d_ff: int = 2048    # paper default
    dropout: float = 0.1
    max_seq_len: int = 128
    share_embeddings: bool = True  # paper default

@dataclass
class TrainingConfig:
    max_epochs: int = 30
    learning_rate: float = 1.0    # base LR for Noam schedule (scaled by LambdaLR)
    warmup_steps: int = 4000      # paper default
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.1  # paper default
    accumulate_grad_batches: int = 2
    optimizer_betas: tuple = (0.9, 0.98)  # paper default
    optimizer_eps: float = 1e-9    # paper default

@dataclass
class Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        # Ensure model and data configs use same max_seq_len
        self.model.max_seq_len = self.data.max_seq_len