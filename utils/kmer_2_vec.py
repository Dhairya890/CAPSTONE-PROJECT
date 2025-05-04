import pandas as pd
import numpy as np
from itertools import product
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score


def get_kmer_list(sequences, k, stride=1):
    """
    Splits each sequence into a list of k-mers.

    Args:
        sequences (List[str]): List of DNA sequences.
        k (int): k-mer size.
        stride (int): Stride for sliding window.

    Returns:
        List[List[str]]: List of k-mer lists.
    """
    return [[seq[i:i+k] for i in range(0, len(seq)-k+1, stride)] for seq in sequences]

def train_word2vec(corpus, embedding_dim = 128, window=8, epochs=10):
    """
    Trains a Word2Vec model on k-mer corpus.

    Args:
        corpus (List[List[str]]): Tokenized corpus.
        embedding_dim (int): Embedding vector size.
        window (int): Context window size.
        epochs (int): Training epochs.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    model = Word2Vec(sentences=corpus, vector_size=embedding_dim, window=window, min_count=3, workers=4, sg=0, epochs=epochs, compute_loss=True, batch_words=10000)
    return model

def build_vocab(k=6):
    """
    Builds vocabulary for all possible k-mers.

    Args:
        k (int): k-mer size.

    Returns:
        dict: Mapping k-mer to index (starting from 1, 0 reserved for padding)
    """
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    return {kmer: idx+1 for idx, kmer in enumerate(kmers)}


def build_embedding_matrix(vocab, pretrained_embeddings, embedding_dim):
    """
    Builds embedding matrix from pretrained embeddings.

    Args:
        vocab (dict): Vocabulary mapping.
        pretrained_embeddings (dict): Embeddings lookup.
        embedding_dim (int): Embedding vector size.

    Returns:
        torch.Tensor: Embedding matrix.
    """
    vocab_size = len(vocab) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for kmer, idx in vocab.items():
        embedding_matrix[idx] = pretrained_embeddings.get(kmer, np.random.normal(scale=0.6, size=(embedding_dim,)))
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Neural Network for Word_2_Vec:

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class SimpleCNN(nn.Module):
    def __init__(self, embedding_matrix, freeze_embed = False):
        """
        Simple CNN model for DNA sequence classification using pre-trained embeddings.

        Args:
            embedding_matrix (torch.Tensor): Pre-trained embedding matrix.
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = freeze_embed, padding_idx=0)

        self.conv1 = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=11, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)

def train_and_evaluate(model, train_loader, valid_loader, test_loader, device, epochs=20, lr=1e-3, weight_decay=1e-4):
    """
    Trains the model and evaluates on validation and test sets.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): DataLoader for training.
        valid_loader (DataLoader): DataLoader for validation.
        test_loader (DataLoader): DataLoader for testing.
        device (torch.device): Device to use.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization.

    Returns:
        Tuple[nn.Module, List[dict]]: Trained model and training history.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = []

    for epoch in range(epochs):
        print(f"🔄 Epoch {epoch+1}/{epochs} started...")
        
        # TRAIN
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds_bin = (torch.sigmoid(preds) >= 0.5).float()
            correct += (preds_bin == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds_list, val_labels_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                probs = torch.sigmoid(logits)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = (probs >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                val_preds_list.append(probs.cpu())
                val_labels_list.append(y_batch.cpu())
        
        avg_val_loss = val_loss / len(valid_loader)
        val_acc = val_correct / val_total
        val_preds_all = torch.cat(val_preds_list)
        val_labels_all = torch.cat(val_labels_list)
        val_roc_auc = roc_auc_score(val_labels_all.numpy(), val_preds_all.numpy())

        print(f"📈 Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")

        history.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'val_roc_auc': val_roc_auc
        })

    # TEST
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    test_preds_list, test_labels_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            loss = criterion(logits, y_batch)
            test_loss += loss.item()

            preds = (probs >= 0.5).float()
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
            test_preds_list.append(probs.cpu())
            test_labels_list.append(y_batch.cpu())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_preds_all = torch.cat(test_preds_list)
    test_labels_all = torch.cat(test_labels_list)
    test_roc_auc = roc_auc_score(test_labels_all.numpy(), test_preds_all.numpy())

    print(f"✅ Final Test | Loss: {avg_test_loss:.4f} | Acc: {test_acc:.4f} | ROC-AUC: {test_roc_auc:.4f}")
    
    test_pr_auc = average_precision_score(test_labels_all.numpy(), test_preds_all.numpy())
    test_metrics = {
        'test_loss': avg_test_loss,
        'test_acc': test_acc,
        'test_roc_auc': test_roc_auc,
        'test_pr_auc': test_pr_auc
    }

    return model, history[-1], test_metrics


def predict_sequence(model, w2v_model, vocab, sequence, k=6, stride=1, max_len=96, device='cpu'):
    # Step 0: Uppercase input
    sequence = sequence.upper()

    # Step 0.5: Validate input
    allowed_chars = {'A', 'C', 'G', 'T'}
    if not set(sequence).issubset(allowed_chars):
        raise ValueError(f"Invalid characters found in sequence. Allowed characters: {allowed_chars}")

    # Step 1: Tokenize input into k-mers
    tokens = [sequence[i:i+k] for i in range(0, len(sequence)-k+1, stride)]

    # Step 2: Map k-mers to indices
    indices = [vocab.get(kmer, 0) for kmer in tokens]  # 0 for unknowns/padding

    # Step 3: Pad or truncate
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # shape [1, seq_len]

    # Step 4: Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        proba = torch.sigmoid(logits).item()

    pred_label = 1 if proba >= 0.5 else 0

    return pred_label, proba
