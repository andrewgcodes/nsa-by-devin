"""Train NSA transformer on Shakespeare dataset."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from nsa.model import NSATransformer
from data_utils import load_shakespeare

def train(
    seq_len: int = 256,  # Match window_size from config
    batch_size: int = 16,  # Smaller batch size for stability
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    head_dim: int = 32,  # Match head_dim from config
    num_epochs: int = 10,
    learning_rate: float = 1e-4,  # Small learning rate for stability
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train NSA transformer on Shakespeare dataset."""
    # Load data
    train_loader, val_loader, vocab_size = load_shakespeare(seq_len, batch_size)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model with specific head dimension
    model = NSATransformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
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
            
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "shakespeare_model.pt")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    train()
