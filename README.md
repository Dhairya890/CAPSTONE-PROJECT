# TF Binding Sites Prediction

## Project Overview
This project aims to predict transcription factor (TF) binding sites from DNA sequence data. Using a subset of the *ENCODE* dataset, we classify 101-base pair DNA sequences as binding or non-binding. We compare multiple models, including **convolutional neural networks (CNNs)** trained on different encoding schemes and a **traditional machine learning model (XGBoost)**. The project is designed to process large-scale genomic data, automate experiments across multiple dataset folders, and store results systematically.

## Project Structure
The repository is organized as follows:


### Folder Details
- **notebooks/**: Contains `.ipynb` files used for exploratory data analysis (EDA), testing encoding techniques, debugging models, and validating intermediate results.
- **models/**: Stores the trained models saved automatically for each dataset folder iteration, allowing checkpointing and per-folder evaluation.
- **output/**: Contains an Excel file summarizing metrics (accuracy, AUC, loss, etc.) for all dataset folders processed. Each row corresponds to a different folder/run.
- **utils/**: Contains modular Python scripts (`.py`) with reusable functions like `one_hot.py`, `kmer.py`, `load_sequence_data.py`, etc., to keep codebase clean and maintainable.

## Methodology
1. **Data Preprocessing**
   - Removed ambiguous DNA bases (e.g., sequences with "N").
   - Applied **one-hot encoding** or **k-mer encoding** (with configurable k and stride).
   
2. **Model Development**
   - Trained CNN models with different configurations (using somewhat manual tuning and *Optuna*).
   - Compared CNNs to an XGBoost classifier trained on **count-based k-mer features** (CountVectorizer / TF-IDF).
   - Automated evaluation across 50 dataset folders (~1.5 million training samples, ~400K testing samples).

3. **Explainability**
   - Used **Optuna visualization tools** (e.g., hyperparameter importance plots) to analyze which hyperparameters and encoding settings most influenced model performance.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Dhairya890/CAPSTONE-PROJECT.git
pip install -r requirements.txt
streamlit run app.py
```

For questions, reach out at dc1626@scarletmail.rutgers.edu
