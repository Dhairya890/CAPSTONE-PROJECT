import pandas as pd

def load_sequence_data(file_path):
    """
    Loads DNA sequences from a file, removing any sequences containing 'N'.

    Args:
        file_path (str): Path to input file with columns: identifier, sequence, label.

    Returns:
        pd.DataFrame: DataFrame with columns ['sequence', 'label']
    """
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['identifier', 'sequence', 'label'])
    
    df = df[~df['sequence'].str.contains('N')]
    
    return df[['sequence', 'label']]
