"""Test Shakespeare dataset loading."""
from data_utils import load_shakespeare
import torch

def main():
    # Test data loading with train/val split
    train_loader, val_loader, vocab_size = load_shakespeare(seq_len=64, batch_size=4)
    print(f'Vocabulary size: {vocab_size}')
    
    # Examine train set
    train_x, train_y = next(iter(train_loader))
    print(f'\nTrain input shape: {train_x.shape}')
    print(f'Train target shape: {train_y.shape}')
    
    # Examine val set
    val_x, val_y = next(iter(val_loader))
    print(f'\nVal input shape: {val_x.shape}')
    print(f'Val target shape: {val_y.shape}')
    
    # Get datasets to examine text
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    print(f'\nTrain sample:\n{train_dataset.decode(train_x[0][:100])}')
    print(f'\nVal sample:\n{val_dataset.decode(val_x[0][:100])}')
    
    # Print vocabulary statistics
    print(f'\nVocabulary size: {vocab_size}')
    print('Sample characters:', ''.join(sorted(list(train_dataset.char_to_idx.keys()))[:20]))

if __name__ == "__main__":
    main()
