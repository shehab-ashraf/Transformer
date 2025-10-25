"""
IWSLT2017 DataModule for German→English Translation
====================================================

Handles everything related to loading and preparing data for training:
- Downloads the IWSLT2017 dataset (German to English)
- Builds a BPE tokenizer on all the data
- Tokenizes sentences into numbers
- Filters out sequences that are too long
- Creates train/val/test dataloaders with proper padding

What is IWSLT2017?
------------------
IWSLT = International Workshop on Spoken Language Translation
A dataset of TED talk transcripts translated from German to English.

What This Module Does:
----------------------
1. prepare(): 
   - Downloads dataset from HuggingFace
   - Builds tokenizer (learns vocabulary from data)
   - Tokenizes all sentences (text → numbers)
   - Filters sequences longer than max_seq_len
   
2. setup():
   - Creates PyTorch datasets for train/val/test
   
3. dataloader():
   - Returns dataloaders
"""

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from typing import Tuple, List

from .build_tokenizer import build_tokenizer
from .BaseDataset import BaseDataset


class IWSLT2017DataModule(LightningDataModule):

    def __init__(
        self, 
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        max_seq_len: int,
        vocab_size: int,
    ):
        """
        Initialize the DataModule with configuration.
        
        Args:
            batch_size (int): Number of translation pairs per batch
                             Typical values: 32-128 (depends on GPU memory)
            num_workers (int): Workers for parallel data loading
            pin_memory (bool): If True, pins memory for faster GPU transfer
                              Set to True when using GPU
            max_seq_len (int): Maximum allowed sequence length
                              Sequences longer than this are filtered out
            vocab_size (int): Size of BPE vocabulary to learn
        """
        super().__init__()
        
        # Save all hyperparameters for checkpoint compatibility
        self.save_hyperparameters()

        # Store configuration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Will be initialized in prepare() and setup()
        self.tokenizer = None
        self.dataset_dict = None
        self.train_ds, self.val_ds, self.test_ds = None, None, None


    def prepare(self) -> None:
        """
        Download dataset, build tokenizer, tokenize, and filter.
        
        This method does the heavy lifting of data preparation:
        1. Downloads IWSLT2017 dataset from HuggingFace
        2. Builds BPE tokenizer from all available text
        3. Tokenizes all sentences (converts text to token IDs)
        4. Filters out sequences that exceed max_seq_len
        """

        ##========================Load Raw Dataset================================##
        print("Loading IWSLT2017 dataset...")
        # Download from HuggingFace hub (cached after first download)
        self.dataset_dict = load_dataset(
            "iwslt2017", 
            "iwslt2017-de-en", 
            trust_remote_code=True
        )
        
        # Show dataset sizes
        print(
            f"Loaded dataset: "
            f"Train={len(self.dataset_dict['train'])}, "
            f"Val={len(self.dataset_dict['validation'])}, "
            f"Test={len(self.dataset_dict['test'])}"
        )

        ##========================Build Tokenizer================================##
        print("Building tokenizer on all splits...")
        # Trains BPE tokenizer on all available text (train + val + test)
        self.tokenizer = build_tokenizer(
            dataset_dict=self.dataset_dict,
            vocab_size=self.vocab_size
        )

        ##========================Tokenize All Splits================================##
        # Convert text to token IDs for each split
        for split_name in ['train', 'validation', 'test']:
            print(f"Tokenizing {split_name} split...")
            # map() applies _tokenize_example to every sample
            self.dataset_dict[split_name] = self.dataset_dict[split_name].map(
                self._tokenize_example,
                desc=f"Tokenizing {split_name}"
            )

        ##========================Filter by Length================================##
        # Remove sequences that are too long
        for split_name in ['train', 'validation', 'test']:
            print(f"Filtering {split_name} split by length ≤ {self.max_seq_len}...")
            
            # First compute lengths (batched for speed)
            self.dataset_dict[split_name] = (
                self.dataset_dict[split_name]
                .map(
                    self._add_lengths, 
                    batched=True, 
                    batch_size=10000, 
                    desc=f"Computing lengths {split_name}"
                )
                # Then filter based on length
                .filter(self._filter_by_length)
            )

        # Show final dataset sizes after filtering
        print(
            f"Final dataset sizes: "
            f"Train={len(self.dataset_dict['train'])}, "
            f"Val={len(self.dataset_dict['validation'])}, "
            f"Test={len(self.dataset_dict['test'])}"
        )

    ##========================HELPER METHODS================================##
    def _tokenize_example(self, sample: dict) -> dict:
        """
        Tokenize a single translation pair.
        
        Converts text sentences to lists of token IDs using the trained tokenizer.
        
        Args:
            sample (dict): Dictionary with structure:
                          {'translation': {'de': str, 'en': str}}
        
        Returns:
            dict: Dictionary with tokenized sequences:
                  {'translation_src': List[int], 'translation_trg': List[int]}
        """
        return {
            'translation_src': self.tokenizer.encode(sample['translation']['de']).ids,
            'translation_trg': self.tokenizer.encode(sample['translation']['en']).ids,
        }

    def _add_lengths(self, batch: dict) -> dict:
        """
        Compute sequence lengths for a batch of samples.
        Used for filtering sequences that exceed max_seq_len.
        
        Args:
            batch (dict): Batch dictionary containing:
                         {'translation_src': List[List[int]],
                          'translation_trg': List[List[int]]}
        
        Returns:
            dict: Same batch with added length fields:
                  {'length_src': List[int], 'length_trg': List[int]}
        """
        return {
            'length_src': [len(sample) for sample in batch['translation_src']],
            'length_trg': [len(sample) for sample in batch['translation_trg']],
        }

    def _filter_by_length(self, sample: dict) -> bool:
        """
        Args:
            sample (dict): Sample with 'length_src' and 'length_trg' fields
        
        Returns:
            bool: True if sample should be kept, False if filtered out
        """
        return (
            sample['length_src'] <= self.max_seq_len and 
            sample['length_trg'] <= self.max_seq_len
        )


    def setup(self, stage: str = None) -> None:
        """
        Create PyTorch datasets from the prepared data.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or None
                                  Not used here (we prepare all splits)
        """

        self.train_ds = BaseDataset(self.dataset_dict['train'])
        self.val_ds = BaseDataset(self.dataset_dict['validation'])
        self.test_ds = BaseDataset(self.dataset_dict['test'])


    def pad_collate_fn(
        self, 
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to pad sequences to same length in a batch.
        Sequences in a batch have different lengths. We pad shorter sequences
        with [PAD] tokens so they can be stacked into a single tensor.
        
        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]): 
                List of (src, trg) tensor pairs from dataset
                Each tensor has shape (seq_len,) with variable seq_len
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded tensors
                - src_padded: (batch_size, max_src_len)
                - trg_padded: (batch_size, max_trg_len)
        """
        # Unzip batch into separate lists of source and target sequences
        src_sequences, trg_sequences = zip(*batch)
        
        # Get padding token ID
        pad_id = self.tokenizer.token_to_id("[PAD]")

        # Pad sequences to same length (longest in batch)
        src_padded = pad_sequence(
            src_sequences, 
            batch_first=True, 
            padding_value=pad_id
        )
        trg_padded = pad_sequence(
            trg_sequences, 
            batch_first=True, 
            padding_value=pad_id
        )

        return src_padded, trg_padded

    ##========================DATALOADER METHODS================================##
    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.
        
        Returns:
            DataLoader: Training data with shuffling and parallel loading
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader.
        
        Returns:
            DataLoader: Validation data without shuffling
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader.
        
        Returns:
            DataLoader: Test data without shuffling
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )