import os
import pandas as pd

def initialize_results_df(data_dir, output_excel_path):
    """
    Initializes a results DataFrame by scanning subfolders and creating:
      1. A new DataFrame with train/test paths
      2. An Excel-compatible DataFrame with placeholders for metrics

    Args:
        data_dir (str): Path to the main data directory containing subfolders.
        output_excel_path (str): Path to the Excel file for storing metrics.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (new_df, excel_df)
    """
    folder_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) ]

    train_paths = [os.path.join(data_dir, folder, 'train.data') for folder in folder_names]
    test_paths = [os.path.join(data_dir, folder, 'test.data') for folder in folder_names]

    new_df = pd.DataFrame({
        'folder_name': folder_names,
        'train_path': train_paths,
        'test_path': test_paths,
    })

    excel_min_df = pd.DataFrame({
        'folder_name': folder_names,
        'train_accuracy': [None] * len(folder_names),
        'test_accuracy': [None] * len(folder_names),
        'pr-roc': [None] * len(folder_names),
        'pr-auc': [None] * len(folder_names),
    })

    if os.path.exists(output_excel_path):
        excel_df = pd.read_excel(output_excel_path)
    else:
        excel_min_df.to_excel(output_excel_path, index=False)
        print(f"Excel file saved at: {output_excel_path}")
        excel_df = excel_min_df

    return new_df, excel_df 

