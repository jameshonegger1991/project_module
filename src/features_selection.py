import numpy as np
from sklearn.feature_selection import VarianceThreshold


def apply_variance_threshold(X_train: np.ndarray, X_test: np.ndarray, all_cols: list[str], numeric_cols: list[str], threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray, list[str], VarianceThreshold]:
    """Apply VarianceThreshold only to the numerical transformed columns."""
    
    if X_train.shape[1] != len(all_cols):
        raise ValueError("X_train column count does not match all_cols: "f"{X_train.shape[1]} != {len(all_cols)}")

    numeric_indices = [index for index, column in enumerate(all_cols) if column in numeric_cols]
    
    if len(numeric_indices) != len(numeric_cols):
        raise ValueError("Some numeric column names were not found in all_cols.")

    X_train_num = X_train[:, numeric_indices]
    X_test_num = X_test[:, numeric_indices]

    selector = VarianceThreshold(threshold=threshold)
    X_train_selected = selector.fit_transform(X_train_num)
    X_test_selected = selector.transform(X_test_num)

    selected_cols = [numeric_cols[index] for index in selector.get_support(indices=True)]

    return X_train_selected, X_test_selected, selected_cols, selector
   