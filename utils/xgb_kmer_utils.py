import time
import itertools
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import xgboost as xgb

def get_kmers_stride(seq, k, stride):
    """
    Splits a DNA sequence into overlapping k-mers with given stride.

    Args:
        seq (str): DNA sequence.
        k (int): k-mer size.
        stride (int): stride step.

    Returns:
        str: Space-separated k-mer string.
    """
    return ' '.join([seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)])

def prepare_data(train_x, test_x, k, stride, use_tfidf=False):
    """
    Converts DNA sequences to k-mer count or tf-idf feature matrices.

    Args:
        train_x (pd.Series): Training sequences.
        test_x (pd.Series): Testing sequences.
        k (int): k-mer size.
        stride (int): stride for sliding window.
        use_tfidf (bool): Whether to use TF-IDF instead of count vectorizer.

    Returns:
        Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]: X_train, X_test matrices.
    """
    train_kmers = train_x.apply(lambda seq: get_kmers_stride(seq, k=k, stride=stride))
    test_kmers = test_x.apply(lambda seq: get_kmers_stride(seq, k=k, stride=stride))

    vectorizer = TfidfVectorizer() if use_tfidf else CountVectorizer()
    X_train = vectorizer.fit_transform(train_kmers)
    X_test = vectorizer.transform(test_kmers)

    return X_train, X_test

def train_evaluate(X_train, train_y, X_test, test_y, params):
    """
    Trains an XGBoost classifier and evaluates accuracy.

    Args:
        X_train (sparse matrix): Training feature matrix.
        train_y (pd.Series): Training labels.
        X_test (sparse matrix): Testing feature matrix.
        test_y (pd.Series): Testing labels.
        params (dict): XGBoost hyperparameters.

    Returns:
        float: Test set accuracy.
    """
    model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        verbosity=0
    )
    model.fit(X_train, train_y)
    preds = model.predict(X_test)
    acc = accuracy_score(test_y, preds)
    return acc

# def run_xgb_grid_search(train_df, test_df, k_values, stride_values, xgb_param_grid, output_csv, use_tfidf=False):
#     """
#     Runs grid search over k, stride, and XGBoost hyperparameters.

#     Args:
#         train_df (pd.DataFrame): Training data with 'sequence' and 'label' columns.
#         test_df (pd.DataFrame): Testing data with 'sequence' and 'label' columns.
#         k_values (List[int]): List of k-mer sizes to test.
#         stride_values (List[int]): List of stride values to test.
#         xgb_param_grid (dict): Grid of XGBoost hyperparameters (each param → list).
#         output_csv (str): Path to save CSV results.
#         use_tfidf (bool): Whether to use TF-IDF instead of count vectorizer.

#     Returns:
#         pd.DataFrame: Results DataFrame sorted by accuracy.
#     """

#     train_x = train_df['sequence']
#     train_y = train_df['label']
#     test_x = test_df['sequence']
#     test_y = test_df['label']

#     results = []
#     start_time = time.time()

#     for k, stride in itertools.product(k_values, stride_values):
#         X_train, X_test = prepare_data(train_x, test_x, k, stride, use_tfidf=use_tfidf)

#         for params in itertools.product(
#             xgb_param_grid['n_estimators'],
#             xgb_param_grid['max_depth'],
#             xgb_param_grid['learning_rate'],
#             xgb_param_grid['subsample'],
#             xgb_param_grid['colsample_bytree']
#         ):
#             param_dict = {
#                 'n_estimators': params[0],
#                 'max_depth': params[1],
#                 'learning_rate': params[2],
#                 'subsample': params[3],
#                 'colsample_bytree': params[4]
#             }

#             acc = train_evaluate(X_train, train_y, X_test, test_y, param_dict)

#             results.append({
#                 'k': k,
#                 'stride': stride,
#                 **param_dict,
#                 'accuracy': acc
#             })

#             print(f"✅ k={k}, stride={stride}, acc={acc:.4f}")

#     end_time = time.time()

#     results_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False)
#     results_df.to_csv(output_csv, index=False)

#     print(f"🎯 Best Config:")
#     print(results_df.head(5))
#     print(f"⚡ Total time taken: {end_time-start_time:.2f} seconds.")

#     return results_df

import random

def run_xgb_random_search(train_df, test_df, k_values, stride_values, xgb_param_grid, output_csv, vectorizer, n_trials=20):
    """
    Runs random search over k, stride, and XGBoost hyperparameters.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data.
        k_values (List[int]): List of k-mer sizes.
        stride_values (List[int]): List of stride values.
        xgb_param_grid (dict): Hyperparameter grid.
        output_csv (str): Path to save CSV results.
        vectorizer: Predefined vectorizer.
        n_trials (int): Number of random trials.

    Returns:
        pd.DataFrame: Results DataFrame.
    """
    
    train_x = train_df['sequence']
    train_y = train_df['label']
    test_x = test_df['sequence']
    test_y = test_df['label']

    # Generate full param grid
    all_combos = list(itertools.product(
        k_values,
        stride_values,
        xgb_param_grid['n_estimators'],
        xgb_param_grid['max_depth'],
        xgb_param_grid['learning_rate'],
        xgb_param_grid['subsample'],
        xgb_param_grid['colsample_bytree'],
        xgb_param_grid['gamma']
    ))

    # Randomly sample
    sampled_combos = random.sample(all_combos, min(n_trials, len(all_combos)))

    results = []
    start_time = time.time()

    for trial_idx, params in enumerate(sampled_combos):
        k, stride, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma = params

        X_train, X_test = prepare_data(train_x, test_x, k, stride, vectorizer)

        model = xgb.XGBClassifier(
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            verbosity=0
        )

        model.fit(X_train, train_y)

        preds = model.predict(X_test)
        acc = accuracy_score(test_y, preds)

        results.append({
            'trial': trial_idx + 1,
            'k': k,
            'stride': stride,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'accuracy': acc
        })

        print(f"✅ Trial {trial_idx+1}/{n_trials}: acc={acc:.4f}")

    end_time = time.time()

    results_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False)
    results_df.to_csv(output_csv, index=False)

    print(f"🎯 Best Config:")
    print(results_df.head(5))
    print(f"⚡ Total time: {end_time-start_time:.2f} seconds.")

    return results_df



## FOR LOOPING OVER FOLDERS ##

from itertools import product
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_kmer_vocab(k):
    """
    Builds a vocabulary dictionary of all possible k-mers for given k.

    Args:
        k (int): k-mer size.

    Returns:
        dict: Mapping from k-mer string to index.
    """
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(kmers)}

def build_vectorizer_from_vocab(vocab_dict, use_tfidf=False):
    """
    Creates a CountVectorizer or TfidfVectorizer using a predefined vocabulary.

    Args:
        vocab_dict (dict): Predefined vocabulary mapping.
        use_tfidf (bool): Whether to use TfidfVectorizer instead of CountVectorizer.

    Returns:
        vectorizer: Vectorizer object ready for .transform().
    """
    if use_tfidf:
        return TfidfVectorizer(vocabulary=vocab_dict)
    else:
        return CountVectorizer(vocabulary=vocab_dict)

def get_kmers_str(seq, k, stride):
    """
    Converts a DNA sequence into a space-separated k-mer string using stride.

    Args:
        seq (str): DNA sequence.
        k (int): k-mer size.
        stride (int): stride step.

    Returns:
        str: Space-separated string of k-mers.
    """
    return ' '.join([seq[i:i+k] for i in range(0, len(seq)-k+1, stride)])
