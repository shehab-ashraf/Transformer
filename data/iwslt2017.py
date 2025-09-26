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
        max_seq_len: int,
        vocab_size: int,
        use_full_dataset: bool = True
    ):
        """
        Args:
            batch_size: Number of samples in each batch
            num_workers: Number of subprocesses for data loading
            pin_memory: Pin memory for faster GPU transfer
            max_seq_len: Maximum sequence length to keep
            vocab_size: Size of tokenizer vocabulary
            use_full_dataset: Whether to use full dataset or subset
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.tokenizer = None
        self.dataset_dict = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare(self) -> None:
        """Prepares data by loading, preprocessing and tokenizing"""
        # Load full dataset with all splits
        logging.info("Loading IWSLT2017 dataset with all splits...")
        self.dataset_dict = load_dataset("iwslt2017", "iwslt2017-de-en", trust_remote_code=True)
        
        logging.info(f"Dataset loaded: {self.dataset_dict}")
        logging.info(f"Train: {len(self.dataset_dict['train'])} samples")
        logging.info(f"Validation: {len(self.dataset_dict['validation'])} samples") 
        logging.info(f"Test: {len(self.dataset_dict['test'])} samples")

        # Build and train tokenizer on ALL data (train + validation + test)
        logging.info("Building tokenizer on all data...")
        self.tokenizer = build_tokenizer(
            dataset_dict=self.dataset_dict,
            vocab_size=self.vocab_size
        )

        # Tokenize all splits
        logging.info("Tokenizing all splits...")
        for split_name in ['train', 'validation', 'test']:
            logging.info(f"Tokenizing {split_name} split...")
            self.dataset_dict[split_name] = self.dataset_dict[split_name].map(
                self._tokenize_example,
                desc=f"Tokenizing {split_name}"
            )

        # Add sequence lengths and filter for all splits
        logging.info("Filtering by sequence length...")
        for split_name in ['train', 'validation', 'test']:
            logging.info(f"Filtering {split_name} split...")
            self.dataset_dict[split_name] = (
                self.dataset_dict[split_name].map(
                    self._add_lengths,
                    batched=True,
                    batch_size=10000,
                    desc=f"Computing lengths for {split_name}"
                )
                .filter(self._filter_by_length)
            )
            
        logging.info("Data preparation complete!")
        logging.info(f"Final dataset sizes:")
        logging.info(f"Train: {len(self.dataset_dict['train'])} samples")
        logging.info(f"Validation: {len(self.dataset_dict['validation'])} samples")
        logging.info(f"Test: {len(self.dataset_dict['test'])} samples")

    def _tokenize_example(self, example: dict) -> dict:
        """Tokenizes a single example. SOS/EOS are added by the tokenizer's post-processor"""
        return {
            'translation_src': self.tokenizer.encode(example['translation']['de']).ids,  # German source
            'translation_trg': self.tokenizer.encode(example['translation']['en']).ids,  # English target
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
        """Sets up train/val/test splits"""
        if stage == "fit":
            # Use the predefined train and validation splits
            self.train_ds = BaseDataset(self.dataset_dict['train'])
            self.val_ds = BaseDataset(self.dataset_dict['validation'])
            
        elif stage == "test":
            # Use the predefined test split
            self.test_ds = BaseDataset(self.dataset_dict['test'])

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
    
    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader"""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )