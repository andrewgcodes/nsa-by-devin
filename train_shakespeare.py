"""Train NSA transformer on Shakespeare dataset."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from nsa.model import NSATransformer
from data_utils import load_shakespeare

def train(
    seq_len: int = 256,
    batch_size: int = 32,
    hidden_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train NSA transformer on Shakespeare dataset."""
    # Load data
    dataloader, vocab_size = load_shakespeare(seq_len, batch_size)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = NSATransformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    ).to(device)
    
    # Training setup
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    train()
