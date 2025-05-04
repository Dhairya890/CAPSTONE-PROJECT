import numpy as np
import pandas as pd

def one_hot_encode_sequences(sequences):
    """
    One-hot encodes a list or Series of DNA sequences.

    DNA bases A, C, G, T are mapped to indices 0-3.
    Any unknown bases are ignored (left as zero vector).

    Args:
        sequences (pd.Series or list): DNA sequences containing only A, C, T, G.

    Returns:
        np.ndarray: One-hot encoded array of shape (num_sequences, seq_length, 4)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequences.iloc[0])  # assumes all sequences are same length
    num_seqs = len(sequences)

    one_hot = np.zeros((num_seqs, seq_len, 4), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            if base in mapping:
                one_hot[i, j, mapping[base]] = 1.0
            else:
                # This shouldn't happen, but just in case...
                pass

    return one_hot
