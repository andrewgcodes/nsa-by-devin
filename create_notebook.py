import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = [
    nbf.v4.new_markdown_cell('''# NSA: Natively trainable Sparse Attention
This notebook demonstrates the NSA implementation on Shakespeare text generation using PyTorch.

## Setup
First, let's install the required dependencies and clone the repository:'''),
    
    nbf.v4.new_code_cell('''!pip install torch tqdm matplotlib
!git clone https://github.com/andrewgcodes/nsa-by-devin.git
%cd nsa-by-devin'''),
    
    nbf.v4.new_markdown_cell('## Import Dependencies'),
    nbf.v4.new_code_cell('''import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from nsa.model import NSATransformer
from data_utils import load_shakespeare'''),
    
    nbf.v4.new_markdown_cell('## Load and Process Data'),
    nbf.v4.new_code_cell('''# Configuration
config = {
    'seq_len': 256,
    'batch_size': 16,
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'head_dim': 32,
    'num_epochs': 10,
    'learning_rate': 1e-4
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
train_loader, val_loader, vocab_size = load_shakespeare(
    seq_len=config['seq_len'],
    batch_size=config['batch_size']
)
print(f'Vocabulary size: {vocab_size}')'''),
    
    nbf.v4.new_markdown_cell('## Create Model'),
    nbf.v4.new_code_cell('''# Initialize model
model = NSATransformer(
    vocab_size=vocab_size,
    num_layers=config['num_layers'],
    hidden_dim=config['hidden_dim'],
    num_heads=config['num_heads'],
    head_dim=config['head_dim']
).to(device)

print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')'''),
    
    nbf.v4.new_markdown_cell('## Training Loop'),
    nbf.v4.new_code_cell('''# Training setup
optimizer = Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(config['num_epochs']):
    # Training
    model.train()
    total_train_loss = 0
    train_batches = 0
    
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
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
    
    print(f'Epoch {epoch + 1}/{config['num_epochs']}')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Val Loss: {avg_val_loss:.4f}')'''),
    
    nbf.v4.new_markdown_cell('## Visualize Results'),
    nbf.v4.new_code_cell('''plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()''')
]

nb['cells'] = cells

# Write the notebook
with open('NSA_Shakespeare.ipynb', 'w') as f:
    nbf.write(nb, f)
