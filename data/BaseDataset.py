import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any

class BaseDataset(Dataset):
    """Dataset hat handles source and target sequences"""
    
    def __init__(self, dataset: Dict[str, Any]):
        """
        Args:
            dataset: HuggingFace dataset containing tokenized translation pairs
        """
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a pair of tokenized source and target sequences
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing:
            - Source sequence tensor
            - Target sequence tensor
        """
        src_encoded = self.dataset[idx]['translation_src']
        trg_encoded = self.dataset[idx]['translation_trg']
        
        return (
            torch.tensor(src_encoded, dtype=torch.long),
            torch.tensor(trg_encoded, dtype=torch.long),
        )