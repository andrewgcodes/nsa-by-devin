"""Train a simple NSA transformer on a sequence prediction task."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nsa.model import NSATransformer


class SequenceDataset(Dataset):
    """Simple sequence prediction dataset."""
    
    def __init__(self, seq_len: int = 16, num_sequences: int = 1000):
        self.sequences = []
        for _ in range(num_sequences):
            # Create sequence where each number is sum of previous two
            seq = [torch.randint(0, 10, (1,)).item() for _ in range(2)]
            for i in range(seq_len - 2):
                seq.append((seq[-1] + seq[-2]) % 10)
            self.sequences.append(torch.tensor(seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]  # Input, target pairs


def train():
    # Create dataset
    dataset = SequenceDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Create model
    model = NSATransformer(
        vocab_size=10,
        num_layers=2,
        hidden_dim=256,
        num_heads=4
    )
    device = torch.device('cpu')
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10
    
    # Training loop
    from tqdm import tqdm
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(
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
