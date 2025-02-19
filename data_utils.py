"""Utilities for loading and preprocessing text datasets."""

import os
import requests
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

def download_shakespeare():
    """Download Shakespeare dataset from Project Gutenberg."""
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()
    
    # Save to disk
    with open("data/shakespeare.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print("Shakespeare dataset downloaded successfully")

class CharDataset(Dataset):
    """Character-level dataset for text generation."""
    
    def __init__(self, text: str, char_to_idx: dict, seq_len: int = 256):
        self.text = text
        self.seq_len = seq_len
        
        # Use provided character mapping
        self.char_to_idx = char_to_idx
        self.idx_to_char = {i: ch for ch, i in char_to_idx.items()}
        self.vocab_size = len(char_to_idx)
        
        # Convert text to indices
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence and target
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert indices back to text."""
        return "".join(self.idx_to_char[idx.item()] for idx in indices)

def clean_shakespeare_text(text: str) -> str:
    """Clean Shakespeare text by removing headers, footers, etc."""
    # Find start of actual content
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    if start_idx != -1:
        text = text[start_idx:]
        
    # Remove end matter
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]
    
    # Remove table of contents
    content_start = text.find("THE SONNETS")
    if content_start != -1:
        text = text[content_start:]
    
    return text.strip()

def create_char_vocab(text: str) -> dict:
    """Create character-to-index mapping from text."""
    chars = sorted(list(set(text)))
    return {ch: i for i, ch in enumerate(chars)}

def load_shakespeare(
    seq_len: int = 256,
    batch_size: int = 32,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, int]:
    """Load Shakespeare dataset and create train/val dataloaders.
    
    Args:
        seq_len: Length of sequences to generate
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        
    Returns:
        Training DataLoader
        Validation DataLoader
        Vocabulary size
    """
    # Download if needed
    if not os.path.exists("data/shakespeare.txt"):
        download_shakespeare()
    
    # Load and clean text
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    text = clean_shakespeare_text(text)
    
    # Create vocabulary from full text
    char_to_idx = create_char_vocab(text)
    
    # Split into train/val
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets with shared vocabulary
    train_dataset = CharDataset(train_text, char_to_idx, seq_len=seq_len)
    val_dataset = CharDataset(val_text, char_to_idx, seq_len=seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, train_dataset.vocab_size
