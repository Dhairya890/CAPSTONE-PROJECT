import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from optuna_cnn_kmer_utils import tokenize_sequence

def pad_sequences(sequences, max_len, pad_value=0):
    """
    Pads tokenized sequences to the same max_len with a pad_value.

    Args:
        sequences (List[List[int]]): List of tokenized sequences.
        max_len (int): Desired max length.
        pad_value (int): Value to pad shorter sequences with.

    Returns:
        np.ndarray: Padded array of shape (num_sequences, max_len)
    """
    
    padded = np.full((len(sequences), max_len), pad_value)
    for i, seq in enumerate(sequences):
        padded[i, :min(len(seq), max_len)] = seq[:max_len]
    return padded

def prepare_kmer_loaders(train_sequences, train_labels, test_sequences, test_labels,
                         vocab, k, stride, max_len, batch_size=128, val_ratio=0.2):
    """
    Generates train_loader, valid_loader, test_loader for k-mer encoded sequences.
    
    Args:
        train_sequences (List[str]): list of DNA sequences for training
        train_labels (np.ndarray): labels for training
        test_sequences (List[str]): list of DNA sequences for testing
        test_labels (np.ndarray): labels for testing
        vocab (dict): k-mer vocabulary
        k (int): k-mer size
        stride (int): stride size for k-mer tokenization
        max_len (int): max length for padded sequences
        batch_size (int): DataLoader batch size
        val_ratio (float): validation split ratio
        
    Returns:
        train_loader, valid_loader, test_loader
    """
    # Tokenize
    train_tokens = [tokenize_sequence(seq, vocab, k, stride) for seq in train_sequences]
    test_tokens = [tokenize_sequence(seq, vocab, k, stride) for seq in test_sequences]
    
    # Pad
    X_train_padded = pad_sequences(train_tokens, max_len)
    X_test_padded = pad_sequences(test_tokens, max_len)
    
    # Split train → train/valid
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_padded, train_labels, test_size=val_ratio, random_state=42)
    
    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(X_train_split), torch.tensor(y_train_split, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_split), torch.tensor(y_val_split, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test_padded), torch.tensor(test_labels, dtype=torch.float32))
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
