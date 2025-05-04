import json
from itertools import product

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import Dataset


def build_kmer_vocab(k=5):
    """
    Builds a vocabulary mapping all possible k-mers of size k to indices.

    Args:
        k (int): k-mer size.

    Returns:
        dict: Mapping from k-mer string to integer index (1-indexed, 0 is padding).
    """
    bases = ["A", "C", "G", "T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
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
    return [
        vocab.get(seq[i : i + k], 0)
        for i in range(0, len(seq) - k + 1, stride)
    ]


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
                tokens = tokens[: self.max_len]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            label, dtype=torch.float32
        )


class DynamicCNN(nn.Module):
    def __init__(self, vocab_size, hp, max_len=None):
        super().__init__()
        self.hp = hp
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hp.get("embedding_dim", 128),
        )

        layers = []
        in_channels = hp.get("embedding_dim", 128)
        for i in range(hp.get("num_layers", 1)):
            out_channels = hp.get(f"units_{i}", 64)
            kernel_size = hp.get(f"kernel_size_{i}", 7)
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds.squeeze(1), y)
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
            outputs = model(x).squeeze(1)
            preds.append(outputs.sigmoid().cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    preds_binary = (preds > 0.5).float()
    acc = (preds_binary == labels).float().mean().item()
    return acc, preds, labels


def objective(
    trial,
    train_loader,
    valid_loader,
    vocab_size,
    device,
    epochs,
    max_len,
    search_space,
):
    hp = {}

    for param in ["num_layers", "embedding_dim"]:
        config = search_space[param]
        if config["type"] == "int":
            hp[param] = trial.suggest_int(param, config["low"], config["high"])
        elif config["type"] == "float":
            hp[param] = trial.suggest_float(
                param,
                config["low"],
                config["high"],
                log=config.get("log", False),
            )
        elif config["type"] == "categorical":
            hp[param] = trial.suggest_categorical(param, config["choices"])

    for i in range(hp["num_layers"]):
        for param in ["units", "kernel_size", "activation", "dropout"]:
            config = search_space[param]
            key = f"{param}_{i}"
            if config["type"] == "int":
                hp[key] = trial.suggest_int(key, config["low"], config["high"])
            elif config["type"] == "float":
                hp[key] = trial.suggest_float(
                    key,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif config["type"] == "categorical":
                hp[key] = trial.suggest_categorical(key, config["choices"])

    model = DynamicCNN(vocab_size, hp, max_len=max_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    acc, preds, labels = evaluate(model, valid_loader, device)
    return acc


def run_optuna_pipeline(
    train_loader,
    valid_loader,
    vocab_size,
    device,
    epochs,
    n_trials,
    max_len,
    save_path,
    search_space,
):
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
    study.optimize(
        lambda trial: objective(
            trial,
            train_loader,
            valid_loader,
            vocab_size,
            device,
            epochs,
            max_len,
            search_space,
        ),
        n_trials=n_trials,
    )

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


def load_optuna_cnn_kmer_model(
    model_path, config_path, vocab_size, device="cpu"
):
    """
    Loads a saved TFBS CNN model and its hyperparameters.

    Args:
        model_path (str): Path to the saved .pt file.
        config_path (str): Path to the saved .json hyperparameter config.
        vocab_size (int): Size of the k-mer vocabulary.
        device (str): 'cpu' or 'cuda'.

    Returns:
        nn.Module: Loaded model.
    """
    with open(config_path, "r") as f:
        hp = json.load(f)
    model = DynamicCNN(vocab_size=vocab_size, hp=hp)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, hp


def predict_optuna_cnn_kmer(
    sequence, model, vocab, k=5, stride=1, max_len=96, device="cpu"
):
    """
    Predicts whether a DNA sequence is a TFBS using the trained CNN model.

    Args:
        sequence (str): Input DNA sequence.
        model (nn.Module): Loaded CNN model.
        vocab (dict): k-mer vocabulary.
        k (int): k-mer size.
        stride (int): stride length used in tokenization.
        max_len (int): maximum input length for the model.
        device (str): 'cpu' or 'cuda'.

    Returns:
        Tuple[str, float]: ("TFBS"/"Non-TFBS", confidence score %)
    """
    tokens = tokenize_sequence(sequence.upper(), vocab, k=k, stride=stride)
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    input_tensor = (
        torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        logits = model(input_tensor).squeeze(1)
        prob = torch.sigmoid(logits).item()

    label = "TFBS" if prob >= 0.5 else "Non-TFBS"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, round(confidence * 100, 2)
