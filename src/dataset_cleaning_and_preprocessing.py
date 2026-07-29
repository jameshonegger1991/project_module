import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def add_derived_features_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Create MMINS and MEAN_ESCS, then remove identifiers/source features."""
    result = df.copy()
    result["MMINS"] = result["ST059Q01TA"] * result["SC175Q01JA"]
    result["MEAN_ESCS"] = result.groupby("CNTSCHID")["ESCS"].transform("mean")
    return result.drop(columns=["ESCS", "ST059Q01TA", "SC175Q01JA", "CNTSCHID", "CNTSTUID"])


def get_missing_percentages_per_column(df: pd.DataFrame) -> pd.Series:
    """Return the percentage of missing values in each column."""
    return df.isnull().mean() * 100


def get_missing_percentages_per_row(df: pd.DataFrame) -> pd.Series:
    """Return the percentage of missing values in each row."""
    return df.isnull().mean(axis=1) * 100


def remove_invalid_rows(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Remove rows with a missing target or excessive feature missingness."""
    if TARGET not in df.columns:
        raise ValueError(f"Target variable '{TARGET}' was not found.")

    result = df.dropna(subset=[TARGET]).copy()
    feature_missingness = get_missing_percentages_per_row(result.drop(columns=[TARGET]))
    return result.loc[feature_missingness < threshold].copy()


def filter_columns_from_training_data(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float,) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select columns from training-set missingness and apply them to test data."""
    missing_pct = get_missing_percentages_per_column(X_train)
    selected_columns = missing_pct[missing_pct < threshold].index.tolist()
    removed_columns = [col for col in X_train.columns if col not in selected_columns]
    
    return (X_train.loc[:, selected_columns].copy(), X_test.loc[:, selected_columns].copy(),removed_columns)


def ordinal_features_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal categorical features to ordered numerical codes."""
   
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


def prepare_data_for_splitting(df: pd.DataFrame, threshold: float = MISSING_VALUES_THRESHOLD) -> tuple:
    """Apply deterministic feature engineering and row-level cleaning."""
    df_raw_with_correct_features = add_derived_features_and_clean(df)
    df_mapped = ordinal_features_mapping(df_raw_with_correct_features)
    df_preprocessed = remove_invalid_rows(df_mapped, threshold)
    df_preprocessed["MMINS"] = df_preprocessed["MMINS"].clip(upper=MMINS_CAP)
    return df_raw_with_correct_features, df_preprocessed


def split_data(df_preprocessed: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple:
    """Split the cleaned dataset into training and test sets."""
    if TARGET not in df_preprocessed.columns:
        raise ValueError(f"Target variable '{TARGET}' is missing before splitting.")
    X = df_preprocessed.drop(columns=[TARGET])
    y = df_preprocessed[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_feature_groups(X_train: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return numerical, categorical and ordinal columns that are still available."""
    categorical_cols = [col for col in CATEGORICAL_COLS if col in X_train.columns]
    ordinal_cols = [col for col in ORDINAL_COLS if col in X_train.columns]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols + ordinal_cols]
    return numeric_cols, categorical_cols, ordinal_cols


def create_imputer_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Create the imputation/encoding preprocessor used before feature selection."""
    numeric_cols, categorical_cols, ordinal_cols = get_feature_groups(X_train)

    numerical_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("ord", ordinal_transformer, ordinal_cols),
        ]
    )


def create_eda_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Create an imputation-only EDA preprocessor retaining every nominal category."""
    numeric_cols, categorical_cols, ordinal_cols = get_feature_groups(X_train)

    numerical_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("ord", ordinal_transformer, ordinal_cols),
        ]
    )


def create_imputed_dataframe(X_imputed: np.ndarray, preprocessor: ColumnTransformer, y: pd.Series) -> pd.DataFrame:
    """Convert a transformed matrix to a labelled DataFrame and append the target."""
    feature_names = preprocessor.get_feature_names_out()
    result = pd.DataFrame(X_imputed, columns=feature_names, index=y.index)
    result[TARGET] = y
    return result.reset_index(drop=True)


def run_preprocessing_pipeline(df: pd.DataFrame, missing_values_threshold: float = MISSING_VALUES_THRESHOLD) -> tuple:
    """Prepare, split, train-filter, impute and encode the PISA dataset."""
    df_raw_with_correct_features, df_preprocessed = prepare_data_for_splitting(df, missing_values_threshold)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    X_train, X_test, removed_missing_columns = filter_columns_from_training_data(X_train, X_test, missing_values_threshold)

    imputer_preprocessor = create_imputer_preprocessor(X_train)
    X_train_imputed = imputer_preprocessor.fit_transform(X_train)
    X_test_imputed = imputer_preprocessor.transform(X_test)

    all_imputed_feature_names = imputer_preprocessor.get_feature_names_out().tolist()
    numeric_imputed_feature_names = [name for name in all_imputed_feature_names if name.startswith("num__")]

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




