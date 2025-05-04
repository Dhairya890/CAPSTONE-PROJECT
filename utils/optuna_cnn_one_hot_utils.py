import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from one_hot import one_hot_encode_sequences
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


def train_model(model, train_loader, valid_loader, device, hp, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            # output = model(batch_x).squeeze()
            output = model(batch_x).squeeze(
                dim=-1
            )  # only squeeze channel/feature dimension
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).squeeze(dim=-1)
            y_true.extend(batch_y.cpu().numpy())
            y_proba.extend(preds.cpu().numpy())
            y_pred.extend((preds.cpu().numpy() > 0.5).astype(int))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


class DynamicCNN(nn.Module):
    def __init__(self, hp, input_len=101):
        super().__init__()
        layers = []
        in_channels = 4
        seq_len = input_len

        for i in range(hp["num_layers"]):
            out_channels = hp[f"units_{i}"]
            kernel_size = hp[f"kernel_size_{i}"]
            activation = hp[f"activation_{i}"]
            pool_size = hp[f"pool_size_{i}"]
            dropout = hp[f"dropout_{i}"]
            dilation = hp.get(f"dilation_{i}", 1)

            conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            )
            layers.append(conv)
            layers.append(nn.BatchNorm1d(out_channels))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())

            layers.append(nn.MaxPool1d(pool_size))
            layers.append(nn.Dropout(dropout))

            in_channels = out_channels
            seq_len = (seq_len - kernel_size + 1) // pool_size
            if seq_len <= 1:
                break

        self.feature_extractor = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 4, input_len)
            out = self.feature_extractor(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_size, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)


def objective(
    trial, train_loader, valid_loader, device, input_len, epochs, search_space
):
    hp = {}

    # global params
    for param in ["num_layers", "lr"]:
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

    # per-layer params
    for i in range(hp["num_layers"]):
        for param in [
            "units",
            "kernel_size",
            "activation",
            "pool_size",
            "dropout",
            "dilation",
        ]:
            config = search_space[param]
            param_key = f"{param}_{i}"
            if config["type"] == "int":
                hp[param_key] = trial.suggest_int(
                    param_key, config["low"], config["high"]
                )
            elif config["type"] == "float":
                hp[param_key] = trial.suggest_float(
                    param_key,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif config["type"] == "categorical":
                hp[param_key] = trial.suggest_categorical(
                    param_key, config["choices"]
                )

    model = DynamicCNN(hp, input_len=input_len)
    model = train_model(
        model, train_loader, valid_loader, device, hp, epochs=epochs
    )
    metrics = evaluate_model(model, valid_loader, device)

    return metrics["average_precision"]


def run_optuna_pipeline(
    train_loader,
    valid_loader,
    device,
    input_len,
    epochs,
    n_trials,
    save_path,
    search_space,
):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial,
            train_loader,
            valid_loader,
            device,
            input_len,
            epochs,
            search_space,
        ),
        n_trials=n_trials,
    )

    best_params = study.best_trial.params
    best_model = DynamicCNN(best_params, input_len=input_len)
    best_model = train_model(
        best_model, train_loader, valid_loader, device, best_params, epochs
    )
    torch.save(best_model.state_dict(), save_path)

    final_metrics = evaluate_model(best_model, valid_loader, device)

    return best_model, best_params, final_metrics, study


def predict_onehot_sequence(model, sequence, device="cuda", seq_len=101):
    """
    Predict whether a DNA sequence is a TFBS using a One-Hot encoded CNN model.

    Args:
        model (torch.nn.Module): Trained CNN model.
        sequence (str): DNA sequence to predict.
        device (str): 'cuda' or 'cpu'.
        seq_len (int): Expected sequence length (will not pad/truncate, assumes fixed input length).

    Returns:
        Tuple[str, float]: ("TFBS" or "Non-TFBS", confidence in %)
    """
    sequence = sequence.upper()
    allowed_chars = {"A", "C", "G", "T"}
    if not set(sequence).issubset(allowed_chars):
        raise ValueError(
            f"Invalid characters found in sequence. Allowed characters: {allowed_chars}"
        )
    if len(sequence) != seq_len:
        raise ValueError(f"Input sequence must be exactly {seq_len} bp long.")

    encoded = one_hot_encode_sequences(pd.Series([sequence]))[
        0
    ]  # shape [seq_len, 4]
    encoded = encoded.transpose(1, 0)  # shape [4, seq_len]
    input_tensor = (
        torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
    )  # shape [1, 4, seq_len]

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # shape [1, 1]
        prob = torch.sigmoid(output).item()  # convert logits to probability

    label = "TFBS" if prob >= 0.5 else "Non-TFBS"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, round(confidence * 100, 2)
