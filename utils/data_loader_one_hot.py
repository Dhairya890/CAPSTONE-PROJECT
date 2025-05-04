import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def prepare_dataloaders(X, y, batch_size=128, val_ratio=0.2):
    """
    Splits the dataset into train/validation and returns DataLoaders.

    Args:
        X (np.ndarray): Input features (one-hot encoded sequences).
        y (pd.Series): Labels (0/1).
        batch_size (int): Batch size for DataLoader.
        val_ratio (float): Fraction for validation set.

    Returns:
        Tuple[DataLoader, DataLoader]: train_loader, val_loader
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train.values))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val.values))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
