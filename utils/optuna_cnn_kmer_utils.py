import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import optuna
import numpy as np
from itertools import product
from torch.utils.data import Dataset

def build_kmer_vocab(k=5):
    """
    Builds a vocabulary mapping all possible k-mers of size k to indices.

    Args:
        k (int): k-mer size.

    Returns:
        dict: Mapping from k-mer string to integer index (1-indexed, 0 is padding).
    """
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    vocab = {kmer: idx + 1 for idx, kmer in enumerate(kmers)}  # +1 for padding
    return vocab

def tokenize_sequence(seq, vocab, k=5, stride=2):
    """
    Tokenizes a DNA sequence into overlapping k-mers mapped by vocab.

    Args:
        seq (str): DNA sequence.
        vocab (dict): k-mer vocabulary.
        k (int): k-mer size.
        stride (int): stride size.

    Returns:
        List[int]: List of integer token ids.
    """
    return [vocab.get(seq[i:i+k], 0) for i in range(0, len(seq)-k+1, stride)]

class PreTokenizedDataset(Dataset):
    def __init__(self, tokenized_seqs, labels, max_len=None):
        self.tokenized_seqs = tokenized_seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.tokenized_seqs)

    def __getitem__(self, idx):
        tokens = self.tokenized_seqs[idx]
        label = self.labels[idx]
        if self.max_len:
            if len(tokens) < self.max_len:
                tokens += [0] * (self.max_len - len(tokens))
            else:
                tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class DynamicCNN(nn.Module):
    def __init__(self, vocab_size, hp, max_len=50):
        """
        PyTorch Dataset for pre-tokenized DNA sequences.

        Each sequence is already tokenized into integer indices (e.g., via k-mer vocab).
        Optionally pads or truncates sequences to max_len.

        Args:
            tokenized_seqs (List[List[int]]): List of tokenized sequences.
            labels (List[int]): Corresponding labels.
            max_len (int, optional): If provided, sequences will be padded/truncated to this length.
        """
        super().__init__()
        embedding_dim = hp['embedding_dim']
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self._current_seq_len = max_len

        layers = []
        in_channels = embedding_dim
        num_layers = hp['num_layers']
        dilation = 1

        for i in range(num_layers):
            out_channels = hp[f'units_{i}']
            kernel_size = hp[f'kernel_size_{i}']
            dropout_rate = hp[f'dropout_{i}']
            activation_name = hp[f'activation_{i}']

            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding='same'))
            if activation_name == "relu":
                layers.append(nn.ReLU())
            elif activation_name == "gelu":
                layers.append(nn.GELU())
            elif activation_name == "silu":
                layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))

            expected_seq_len = (self._current_seq_len + 1) // 2
            if expected_seq_len >= 10:
                layers.append(nn.MaxPool1d(kernel_size=2))
                self._current_seq_len = expected_seq_len

            in_channels = out_channels
            if (i + 1) % 2 == 0:
                dilation = min(dilation * 2, 8)

        self.conv_layers = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds.append(outputs.sigmoid().cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    preds_binary = (preds > 0.5).float()
    acc = (preds_binary == labels).float().mean().item()
    return acc, preds, labels

def objective(trial, train_loader, valid_loader, vocab_size, device, epochs, max_len, search_space):
    hp = {}

    for param in ['num_layers', 'embedding_dim']:
        config = search_space[param]
        if config['type'] == 'int':
            hp[param] = trial.suggest_int(param, config['low'], config['high'])
        elif config['type'] == 'float':
            hp[param] = trial.suggest_float(param, config['low'], config['high'], log=config.get('log', False))
        elif config['type'] == 'categorical':
            hp[param] = trial.suggest_categorical(param, config['choices'])

    for i in range(hp['num_layers']):
        for param in ['units', 'kernel_size', 'activation', 'dropout']:
            config = search_space[param]
            key = f'{param}_{i}'
            if config['type'] == 'int':
                hp[key] = trial.suggest_int(key, config['low'], config['high'])
            elif config['type'] == 'float':
                hp[key] = trial.suggest_float(key, config['low'], config['high'], log=config.get('log', False))
            elif config['type'] == 'categorical':
                hp[key] = trial.suggest_categorical(key, config['choices'])

    model = DynamicCNN(vocab_size, hp, max_len=max_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    acc, preds, labels = evaluate(model, valid_loader, device)
    return acc

def run_optuna_pipeline(train_loader, valid_loader, vocab_size, device, epochs, n_trials, max_len, save_path, search_space):
    """
    Runs an Optuna hyperparameter optimization pipeline.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        vocab_size (int): Size of k-mer vocabulary.
        device (torch.device): Computation device.
        epochs (int): Number of epochs for training each trial.
        n_trials (int): Number of Optuna trials to run.
        max_len (int): Maximum sequence length (for CNN input).
        save_path (str): Path to save the best model weights.
        search_space (dict): Dictionary defining hyperparameter search space.

    Returns:
        Tuple[nn.Module, dict, float, optuna.Study]: 
            (best_model, best_params, best_accuracy, optuna_study_object)
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_loader, valid_loader, vocab_size, device, epochs, max_len, search_space), n_trials=n_trials)

    best_params = study.best_trial.params
    best_model = DynamicCNN(vocab_size, best_params, max_len=max_len)
    best_model.to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_one_epoch(best_model, train_loader, optimizer, criterion, device)

    torch.save(best_model.state_dict(), save_path)
    acc, preds, labels = evaluate(best_model, valid_loader, device)

    return best_model, best_params, acc, study


