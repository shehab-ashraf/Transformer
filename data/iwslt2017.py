from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from typing import Tuple, List

from .build_tokenizer import build_tokenizer
from .BaseDataset import BaseDataset


class IWSLT2017DataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the IWSLT2017 DE→EN translation dataset.
    Handles loading, tokenization, filtering, and batching.
    """

    def __init__(
        self, 
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        max_seq_len: int,
        vocab_size: int,
    ):
        """
        Args:
            batch_size: Number of samples per batch.
            num_workers: Subprocesses for dataloading.
            pin_memory: Faster GPU transfer if True.
            max_seq_len: Maximum allowed sequence length (src and trg).
            vocab_size: Size of tokenizer vocabulary.
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
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare(self) -> None:
        """
        Load raw dataset, build tokenizer, tokenize all splits,
        and filter by maximum sequence length.
        """
        print("Loading IWSLT2017 dataset...")
        self.dataset_dict = load_dataset("iwslt2017", "iwslt2017-de-en", trust_remote_code=True)
        print(
            f"Loaded dataset: Train={len(self.dataset_dict['train'])}, "
            f"Val={len(self.dataset_dict['validation'])}, Test={len(self.dataset_dict['test'])}"
        )

        # Build tokenizer
        print("Building tokenizer on all splits...")
        self.tokenizer = build_tokenizer(
            dataset_dict=self.dataset_dict,
            vocab_size=self.vocab_size
        )

        # Tokenize each split
        for split_name in ['train', 'validation', 'test']:
            print(f"Tokenizing {split_name} split...")
            self.dataset_dict[split_name] = self.dataset_dict[split_name].map(
                self._tokenize_example,
                desc=f"Tokenizing {split_name}"
            )

        # Add lengths and filter
        for split_name in ['train', 'validation', 'test']:
            print(f"Filtering {split_name} split by length ≤ {self.max_seq_len}...")
            self.dataset_dict[split_name] = (
                self.dataset_dict[split_name]
                .map(self._add_lengths, batched=True, batch_size=10000, desc=f"Computing lengths {split_name}")
                .filter(self._filter_by_length)
            )

        print(
            f"Final dataset sizes: Train={len(self.dataset_dict['train'])}, "
            f"Val={len(self.dataset_dict['validation'])}, Test={len(self.dataset_dict['test'])}"
        )

    # ---------------------------
    # Helpers
    # ---------------------------

    def _tokenize_example(self, sample: dict) -> dict:
        """Tokenize source (DE) and target (EN) sentences."""
        return {
            'translation_src': self.tokenizer.encode(sample['translation']['de']).ids,
            'translation_trg': self.tokenizer.encode(sample['translation']['en']).ids,
        }

    def _add_lengths(self, batch: dict) -> dict:
        """Add sequence lengths for filtering."""
        return {
            'length_src': [len(sample) for sample in batch['translation_src']],
            'length_trg': [len(sample) for sample in batch['translation_trg']],
        }

    def _filter_by_length(self, sample: dict) -> bool:
        """Keep examples where both src/trg are within max length."""
        return (sample['length_src'] <= self.max_seq_len and 
                sample['length_trg'] <= self.max_seq_len)

    # ---------------------------
    # Lightning Hooks
    # ---------------------------

    def setup(self, stage: str) -> None:
        """Prepare datasets for training/validation/testing."""
        if stage == "fit":
            self.train_ds = BaseDataset(self.dataset_dict['train'])
            self.val_ds = BaseDataset(self.dataset_dict['validation'])
        elif stage == "test":
            self.test_ds = BaseDataset(self.dataset_dict['test'])

    def pad_collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for padding variable-length sequences in a batch.
        Pads both source and target sequences with [PAD].
        """
        src_sequences, trg_sequences = zip(*batch)
        pad_id = self.tokenizer.token_to_id("[PAD]")

        src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=pad_id)
        trg_padded = pad_sequence(trg_sequences, batch_first=True, padding_value=pad_id)

        return src_padded, trg_padded

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_collate_fn,
        )
