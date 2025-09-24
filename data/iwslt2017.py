from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from typing import Tuple, List
import logging

from .build_tokenizer import build_tokenizer
from .BaseDataset import BaseDataset


class IWSLT2017DataModule(LightningDataModule):
    """
    PyTorch Lightning data module for IWSLT2017 DE-EN translation dataset
    """

    def __init__(
        self, 
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        data_size: int,
        max_seq_len: int,
        test_proportion: float,
        vocab_size: int
    ):
        """
        Args:
            batch_size: Number of samples in each batch
            num_workers: Number of subprocesses for data loading
            pin_memory: Pin memory for faster GPU transfer
            data_size: Number of samples to use from dataset
            max_seq_len: Maximum sequence length to keep
            test_proportion: Proportion of data to use for validation
            vocab_size: Size of tokenizer vocabulary
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_size = data_size
        self.max_seq_len = max_seq_len
        self.test_proportion = test_proportion
        self.vocab_size = vocab_size
        
        self.tokenizer = None
        self.data = None

    def prepare(self) -> None:
        """Prepares data by loading, preprocessing and tokenizing"""
        # Load and preprocess dataset
        logging.info("Loading IWSLT2017 dataset...")
        self.data = (
            load_dataset("iwslt2017", "iwslt2017-de-en", split='train', trust_remote_code=True)
            .shuffle(seed=42)
            .select(range(self.data_size))
            .flatten()
        )
        
        # Rename columns for clarity
        self.data = self.data.rename_columns({
            'translation.de': 'translation_trg',
            'translation.en': 'translation_src'
        })

        # Build and train tokenizer
        logging.info("Building tokenizer...")
        self.tokenizer = build_tokenizer(
            data=self.data,
            vocab_size=self.vocab_size
        )

        # Tokenize all sequences
        logging.info("Tokenizing sequences...")
        self.data = self.data.map(
            self._tokenize_example,
            desc="Tokenizing"
        )

        # Add sequence lengths and filter
        logging.info("Filtering by sequence length...")
        self.data = (
            self.data.map(
                self._add_lengths,
                batched=True,
                batch_size=10000,
                desc="Computing lengths"
            )
            .filter(self._filter_by_length)
        )

    def _tokenize_example(self, example: dict) -> dict:
        """Tokenizes a single example. SOS/EOS are added by the tokenizer's post-processor"""
        return {
            'translation_src': self.tokenizer.encode(example['translation_src']).ids,
            'translation_trg': self.tokenizer.encode(example['translation_trg']).ids,
        }

    def _add_lengths(self, example: dict) -> dict:
        """Adds sequence length information to examples"""
        return {
            'length_src': [len(item) for item in example['translation_src']],
            'length_trg': [len(item) for item in example['translation_trg']],
        }

    def _filter_by_length(self, example: dict) -> bool:
        """Filters sequences by maximum allowed length"""
        return (example['length_src'] <= self.max_seq_len and 
                example['length_trg'] <= self.max_seq_len)

    def setup(self, stage: str) -> None:
        """Sets up train/val splits"""
        if stage == "fit":
            # Create train/validation split
            splits = self.data.train_test_split(
                test_size=self.test_proportion,
                seed=42
            )

            # Sort each split by length
            # splits['train'] = splits['train'].sort('length_src', reverse=True)
            # splits['test'] = splits['test'].sort('length_src', reverse=True)

            self.train_ds = BaseDataset(splits['train'])
            self.val_ds = BaseDataset(splits['test'])

    def pad_collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function that handles padding for variable length sequences
        
        Args:
            batch: List of (source, target) tensor pairs
            
        Returns:
            Tuple of padded source and target sequences
        """
        src_sequences, trg_sequences = zip(*batch)
        
        pad_id = self.tokenizer.token_to_id("[PAD]")  # Should be 0
        
        # Pad sequences to max length in batch
        src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=pad_id)
        trg_padded = pad_sequence(trg_sequences, batch_first=True, padding_value=pad_id)
        
        return src_padded, trg_padded

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader"""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader"""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )