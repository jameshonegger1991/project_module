import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    KNN_NEIGHBORS,
    MISSING_VALUES_THRESHOLD,
    MMINS_CAP,
    ORDINAL_COLS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def add_derived_features_and_clean(df):
    """
    Adds derived features (MMINS, MEAN_ESCS) and removes unused columns (identifiers, source).

    Arguments:
    - df: Input DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DF with new features.
    """
    result = df.copy()
    result["MMINS"] = result["ST059Q01TA"] * result["SC175Q01JA"]
    result["MEAN_ESCS"] = result.groupby("CNTSCHID")["ESCS"].transform("mean")
    return result.drop(columns=["ESCS", "ST059Q01TA", "SC175Q01JA", "CNTSCHID", "CNTSTUID"])

def get_missing_percentages_per_column(df):
    """
    Returns the percentage of missing values per column.

    Arguments:
    - df: Input DataFrame.

    Returns:
    - pd.Series: Missing percentage per column.
    """
    return df.isnull().mean() * 100

def get_missing_percentages_per_row(df):
    """
    Returns the percentage of missing values per row.

    Arguments:
    - df: Input DataFrame.

    Returns:
    - pd.Series: Missing % per row.
    """
    return df.isnull().mean(axis=1) * 100

def remove_invalid_rows(df, threshold):
    """
    Removes rows with missing target OR with missing feature % >= threshold.

    Argumentss:
    - df: Input DataFrame.
    - threshold: Maximum allowed missing rate per row.

    Returns:
    - pd.DataFrame: Filtered dataframe.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target variable '{TARGET}' was not found.")

    result = df.dropna(subset=[TARGET]).copy()
    feature_missingness = get_missing_percentages_per_row(result.drop(columns=[TARGET]))
    return result.loc[feature_missingness < threshold].copy()

def filter_columns_from_training_data(X_train, X_test, threshold):
    """
    Drops columns with missing % >= threshold based on training set, applies same to test set.

    Arguments:
    - X_train, X_test: Train/test features.
    - threshold: Maxmimum allowed missing percentage per column.

    Returns:
        tuple: containing the train set and the test set filtered, and the list of removed columns..
    """
    missing_pct = get_missing_percentages_per_column(X_train)
    selected_columns = missing_pct[missing_pct < threshold].index.tolist()
    removed_columns = [col for col in X_train.columns if col not in selected_columns]
    
    return (X_train.loc[:, selected_columns].copy(), X_test.loc[:, selected_columns].copy(),removed_columns)

def ordinal_features_mapping(df):
    """
    Maps ordinal categorical features to ordered numerical values in a given dataframe.
    It returns the input dataframe with encoded ordinal features.

    Arguments:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DF with ordinal features encoded as numbers.
    """
    result = df.copy()
    result["REPEAT"] = result["REPEAT"].replace({"Never repeated": 0, "Repeated at least once": 1, "Repeated at lease once": 1,})

    result["ST062Q01TA"] = result["ST062Q01TA"].replace({
        "Never": 1,
        "One or two times": 2,
        "Three or four times": 3,
        "Five or more times": 4,
    })

    result["IMMIG"] = result["IMMIG"].replace({
        "Native student": 1,
        "Second-Generation student": 2,
        "First-Generation student": 3,
    })

    return result

def prepare_data_for_splitting(df, threshold = MISSING_VALUES_THRESHOLD):
    """
    Applies feature engineering, ordinal mapping, row filtering, and MMINS capping before split.

    Argumentss:
        df: Original DataFrame.
        threshold: Max missing percentage per row. Defaults to MISSING_VALUES_THRESHOLD (= 70%, based on the good practices from the literature).

    Returns:
        tuple: (engineered_df, cleaned_df_ready_for_split)
    """
    df_raw_with_correct_features = add_derived_features_and_clean(df)
    df_mapped = ordinal_features_mapping(df_raw_with_correct_features)
    df_preprocessed = remove_invalid_rows(df_mapped, threshold)
    df_preprocessed["MMINS"] = df_preprocessed["MMINS"].clip(upper=MMINS_CAP)
    return df_raw_with_correct_features, df_preprocessed

def split_data(df_preprocessed, test_size = TEST_SIZE, random_state = RANDOM_STATE):
    """
    Splits cleaned data into train/test sets.

    Arguments:
        df_preprocessed: Cleaned dataframe with target, returned by prepare_data_for_splitting()
        test_size: Proportion for testing. Defaults to TEST_SIZE (= 20%)
        random_state: Random seed. Defaults to RANDOM_STATE.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if TARGET not in df_preprocessed.columns:
        raise ValueError(f"Target variable '{TARGET}' is missing before splitting.")
    X = df_preprocessed.drop(columns=[TARGET])
    y = df_preprocessed[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_feature_groups(X_train):
    """
    Separates features into numerical, categorical, and ordinal groups. It returns a tuple of lists for
    every type of features.

    Arguments:
        X_train: Training features.

    Returns:
        tuple: (numerical_cols, categorical_cols, ordinal_cols)
    """
    categorical_cols = [col for col in CATEGORICAL_COLS if col in X_train.columns]
    ordinal_cols = [col for col in ORDINAL_COLS if col in X_train.columns]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols + ordinal_cols]
    return numeric_cols, categorical_cols, ordinal_cols

def create_imputer_preprocessor(X_train):
    """
    Creates a ColumnTransformer that applies the appropriate preprocessing steps
    to numerical, categorical, and ordinal features

    Arguments:
        X_train: Training features (to identify feature groups).

    Returns:
        ColumnTransformer: a preprocessor applying KNN imputation to numerical features,
        most-frequent imputation and one-hot encoding to categorical features, and
        most-frequent imputation to ordinal features.
    """
    _, categorical_cols, ordinal_cols = get_feature_groups(X_train)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("ord", ordinal_transformer, ordinal_cols),
        ]
    )

def create_eda_preprocessor(X_train):
    """
    Creates ColumnTransformer specifically designed for EDA: KNN imputation for numerical,
    imputation + one-hot (no drop) for categorical, imputation for ordinal. 
    Unlike create_imputer_preprocessor(), no categorical category is dropped, 
    so all categories remain available for exploratory
    analysis and correlation analysis.

    Arguments:
        X_train: Training features.

    Returns:
        ColumnTransformer: Preprocessor for EDA.
    """
    _, categorical_cols, ordinal_cols = get_feature_groups(X_train)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("ord", ordinal_transformer, ordinal_cols),
        ]
    )

def create_imputed_dataframe(X_imputed: np.ndarray, preprocessor: ColumnTransformer, y):
    """
    Converts imputed/encoded matrix back to DataFrame and adds target.

    Arguments:
        X_imputed: Transformed feature matrix.
        preprocessor: Fitted preprocessor (to recover feature names).
        y: Target values.

    Returns:
        pd.DataFrame: Dataframe with transformed features + target.
    """
    feature_names = preprocessor.get_feature_names_out()
    result = pd.DataFrame(X_imputed, columns=feature_names, index=y.index)
    result[TARGET] = y
    return result.reset_index(drop=True)

def numerical_knn_imputation_with_scaling(X_train_num, X_test_num, n_neighbors=KNN_NEIGHBORS):
    """
    Impute numerical features with KNN after scaling, and then restore them to original scale.s
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)
    imputer = KNNImputer(n_neighbors=n_neighbors)

    X_train_imputed_scaled = imputer.fit_transform(X_train_scaled)
    X_test_imputed_scaled = imputer.transform(X_test_scaled)

    X_train_imputed = scaler.inverse_transform(X_train_imputed_scaled)
    X_test_imputed = scaler.inverse_transform(X_test_imputed_scaled)

    return X_train_imputed, X_test_imputed

def preprocessing_training_data_for_EDA(X_train):
    """
    Impute predictive features with appropriate related methods and return combined array with feature names.
    """
    numeric_cols, _, _ = get_feature_groups(X_train)

    X_train_num_imputed, _ = numerical_knn_imputation_with_scaling(X_train[numeric_cols], X_train[numeric_cols], n_neighbors=KNN_NEIGHBORS)
    numeric_feature_names = [f"num__{col}" for col in numeric_cols]

    eda_preprocessor = create_eda_preprocessor(X_train)
    X_train_cat_ord = (eda_preprocessor.fit_transform(X_train))
    cat_ord_feature_names = (eda_preprocessor.get_feature_names_out().tolist())

    X_train_imputed_for_eda = np.hstack([X_train_num_imputed, X_train_cat_ord])
    feature_names = (numeric_feature_names+ cat_ord_feature_names)

    return X_train_imputed_for_eda, feature_names

def run_preprocessing_pipeline(df, missing_values_threshold = MISSING_VALUES_THRESHOLD):
    """
    Runs the full preprocessing pipeline: cleaning, split, column filtering,
    imputation + encoding (fitted on train, applied to test).

    Arguments:
        df: Original DataFrame.
        missing_values_threshold: Missing percentage cutoff for row/column filtering.
        Default set to to MISSING_VALUES_THRESHOLD (70%).

    Returns:
        tuple: (X_train_processed, X_test_processed, X_train_orig, X_test_orig,
                y_train, y_test, df_train_cleaned, df_test_cleaned, feature_names, removed_cols)
    """
    df_raw_with_correct_features, df_preprocessed = prepare_data_for_splitting(df, missing_values_threshold)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    X_train, X_test, removed_missing_columns = filter_columns_from_training_data(X_train, X_test, missing_values_threshold)

    numeric_cols, _, _ = get_feature_groups(X_train)
    X_train_num_imputed, X_test_num_imputed = numerical_knn_imputation_with_scaling(X_train[numeric_cols], X_test[numeric_cols], n_neighbors=KNN_NEIGHBORS)
    numeric_imputed_feature_names = [f"num__{col}" for col in numeric_cols]

    imputer_preprocessor = create_imputer_preprocessor(X_train)
    X_train_cat_ord = imputer_preprocessor.fit_transform(X_train)
    X_test_cat_ord = imputer_preprocessor.transform(X_test)
    cat_ord_feature_names = imputer_preprocessor.get_feature_names_out().tolist()
    X_train_imputed = np.hstack([X_train_num_imputed, X_train_cat_ord])
    X_test_imputed = np.hstack([X_test_num_imputed, X_test_cat_ord])
    all_imputed_feature_names = numeric_imputed_feature_names + cat_ord_feature_names

    return (
        X_train_imputed,
        X_test_imputed,
        X_train,
        X_test,
        y_train,
        y_test,
        df_raw_with_correct_features,
        df_preprocessed,
        numeric_imputed_feature_names,
        all_imputed_feature_names,
        removed_missing_columns,
    )




