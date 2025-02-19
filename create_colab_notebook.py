import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
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
        
        nbf.v4.new_markdown_cell('## Configuration'),
        nbf.v4.new_code_cell('''config = {
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
print(f'Using device: {device}')'''),
        
        nbf.v4.new_markdown_cell('## Load and Process Data'),
        nbf.v4.new_code_cell('''train_loader, val_loader, vocab_size = load_shakespeare(
    seq_len=config['seq_len'],
    batch_size=config['batch_size']
)
print(f'Vocabulary size: {vocab_size}')'''),
        
        nbf.v4.new_markdown_cell('## Training Loop'),
        nbf.v4.new_code_cell('''# Training setup
optimizer = Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(config['num_epochs']):
    model.train()
    total_train_loss = 0
    train_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        train_batches += 1
    
    avg_train_loss = total_train_loss / train_batches
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")''')
    ]
    
    nb['cells'] = cells
    return nb

if __name__ == '__main__':
    notebook = create_notebook()
    with open('NSA_Shakespeare.ipynb', 'w') as f:
        nbf.write(notebook, f)
