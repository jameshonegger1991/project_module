import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
import numpy as np


def add_MEAN_ESCS_and_MMINS_features(df: pd.DataFrame)-> pd.DataFrame :
    """
    Adds 'MEAN_ESCS' and 'MMINS' features to the DataFrame and removes the original
    features used for their creation, along with identifier columns.

    'MMINS' (Mathematics Minutes) is calculated as the product of 'ST059Q01TA'
    (Number of math periods per week) and 'SC175Q01JA' (Average duration of a single math period).
    'MEAN_ESCS' (Mean Economic, Social, and Cultural Status) is calculated as the
    average 'ESCS' for each school ('CNTSCHID').

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.

    This function adds two new features to the dataset: MEAN_ESCS and MMINS. 
    It also removes identifier features (CNTSTUID and CNTSCHID) and the features used to build MEAN_ESCS and MMINS,
    to reduce bias and multicolinearity during training.
    """
    df['MMINS'] = df['ST059Q01TA'] * df['SC175Q01JA']
    df['MEAN_ESCS'] = df.groupby('CNTSCHID')['ESCS'].transform('mean')

    #Original features used to build MEAN_ESCS/MMINS indexes and non-significative features are removed. 
    df = df.drop(columns=['ESCS', 'ST059Q01TA', 'SC175Q01JA', 'CNTSCHID','CNTSTUID'])

    return df

def get_missing_percentages_per_column(df: pd.DataFrame) -> pd.Series:
    """
    Computes the percentage of missing values for each column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A Series where the index is the column name and the values
                   are the percentage of missing values for that column.
    """
    return (df.isnull().sum() / len(df)) * 100

def get_missing_percentages_per_row(df: pd.DataFrame) -> pd.Series:
    """
    Computes the percentage of missing values for each row in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A Series where the index is the row index and the values
                   are the percentage of missing values for that row.
    """
    return (df.isnull().sum(axis=1) / len(df.columns)) * 100

def missing_values_removal(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Removes columns and rows with a percentage of missing values exceeding a given threshold.
    Also removes rows where the target variable 'PV1MATH' is missing.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The percentage threshold (e.g., 70.0) for missing values.
    """
    # 1. Delete columns with missing values >= 70% (based on Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024))
    pct_columns = get_missing_percentages_per_column(df)
    columns_to_keep = pct_columns[pct_columns < threshold].index
    df_step1 = df[columns_to_keep]

    # 2. Delete lines with missing values >= 70% (based on Gamazo & Martínez-Abad (2020), Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024))
    pct_rows = get_missing_percentages_per_row(df_step1)
    df_step2 = df_step1[pct_rows < threshold]

    # 3. Delete lines with missing value in target variable (PV1MATH)
    if 'PV1MATH' not in df_step2.columns:
        raise ValueError("Target variable 'PV1MATH' not found in the DataFrame after column removal.")
    df_cleaned = df_step2.dropna(subset=['PV1MATH'])

    return df_cleaned

def ordinal_features_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps specific ordinal categorical features to numerical codes.
    This function uses `replace()` which handles NaN values gracefully.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    """
    Map ordinal variables to numeric codes. replace() is preferred to map() because it handles NaN values.
    """
    df['REPEAT'] = df['REPEAT'].replace({
        'Never repeated': 0,
        'Repeated at lease once': 1
    })
    
    df['ST062Q01TA'] = df['ST062Q01TA'].replace({
        'Never': 1,
        'One or two times': 2,
        'Three or four times': 3,
        'Five or more times': 4
    })
    
    df['IMMIG'] = df['IMMIG'].replace({
        'Native student': 1,
        'Second-Generation student': 2,
        'First-Generation student': 3
    })
    
    return df

def data_preprocessing_pipeline(df: pd.DataFrame, threshold: float) -> tuple:
    """
    Executes a comprehensive data preprocessing pipeline, ensuring strict data leakage prevention.

    The pipeline involves:
    1. Initial Filtering:
       - Adds 'MEAN_ESCS' and 'MMINS' features.
       - Maps ordinal features to numerical codes.
       - Removes rows with missing target variable ('PV1MATH') and columns/rows
         exceeding a specified missing value threshold.
       - Caps 'MMINS' to a plausible maximum value.

    1. Train/Test Split:
       - The filtered dataset is split into train (80%) and test (20%) sets.
       
    3. Transformation Pipeline (on the TRAINING set):
       - Imputation: Remaining missing values are imputed using the median for numeric features, 
         and the mode for categorical features.
       - Nominal features (ST004D01T, SCHLTYPE) are one-hot encoded with drop='first' 
         to avoid multicollinearity.
       - Ordinal (ST062Q01TA, IMMIG, REPEAT) are preserved as numeric to maintain 
         their hierarchical order.
       - All numeric features are transformed using Yeo-Johnson to correct skewness, then standardised using Z-score scaling (StandardScaler). This ensures comparability across all models (including MLR, SVR, and KNN).

    Parameters:
        df (pd.DataFrame): The raw input DataFrame.
        threshold (float): The missing value percentage threshold for column and row removal.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Test features.
            - y_train (pd.Series): Training target variable.
            - y_test (pd.Series): Test target variable.
            - df_raw_with_correct_features (pd.DataFrame): DataFrame after initial feature engineering.
            - df_preprocessed (pd.DataFrame): DataFrame after missing value handling and MMINS capping.
            - preprocessor (ColumnTransformer): The fitted ColumnTransformer for data transformation.
    """
    #1. Initial Filtering

    # dataset features correspond to the ones described in project report (MEAN_ESCS + MMINS added, intermediate/identifier features removed)
    df_raw_with_correct_features = add_MEAN_ESCS_and_MMINS_features(df)

    # Ordinal features (IMMIG, 'ST062Q01TA', 'REPEAT') are mapped in numeric format to preserve order and scale.
    df_mapped = ordinal_features_mapping(df_raw_with_correct_features.copy())

    # Rows with no PV1MATH value are removed. Rows and columns >=70% missing values are removed.
    df_preprocessed = missing_values_removal(df_mapped, threshold)

    # "MMINS" presents a very strong skewness. To avoid excessive outliers (implausible values), MMINS is capped at 450 minutes.
    # This threshold is based on the structure of the Swiss education system:
    # a standard mathematics period lasts 45 minutes, with a maximum of approximately 10 periods per week.
    # This includes the compulsory mathematics course (5 periods for every student) plus the optional "Mathematics and Physics" course (4-5 additional periods).
    # SOURCE: https://edk.ch/en/education-system 
    df_preprocessed['MMINS'] = df_preprocessed['MMINS'].clip(upper=450)


    #2. Train/Test Split
    if 'PV1MATH' not in df_preprocessed.columns:
        raise ValueError("Target variable 'PV1MATH' is missing from the DataFrame before train/test split.")
    
    X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop(columns = ['PV1MATH']), df_preprocessed['PV1MATH'], test_size=0.2, random_state = 7) # Following common practice in the literature, a 20% test set is used.

    #3. Transformation Pipeline
    # Define feature types for the ColumnTransformer
    categorical_features = ['ST004D01T', 'SCHLTYPE']
    ordinal_features = ['ST062Q01TA', 'REPEAT', 'IMMIG'] 
    numerical_features = [col for col in X_train.columns if col not in categorical_features + ordinal_features] 

    # Yeo-Johnson corrects skewness in numerical features ('PAREDINT', 'PROATCE', 'STUBEHA') to satisfy 
    # the normality assumption of MLR and SVR. StandardScaler then ensures equal scaling across 
    # all features, which is required for distance-based models. 
    # Only the numerical features have a strong percentage of null values. Therefore, the KNN imputation technique is required.
    # A single pipeline is applied to all models to maintain valid inter-model SHAP comparisons.
    # Sources: - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    #          - https://machinelearningmastery.com/power-transforms-with-scikit-learn/ 
    #          - https://medium.com/@tarangds/a-comprehensive-guide-to-data-imputation-techniques-strategies-and-best-practices-152a10fee543  
    
    # Numerical features: Impute with median, apply Yeo-Johnson transformation, then scale. 
    numerical_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=8)),('yeo_johnson', PowerTransformer(method='yeo-johnson')),('scaler', StandardScaler())]) # n_neighbors = 8 : based on the recommandations made by Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024). Applying Machine Learning and SHAP Method to Identify Key Influences on Middle-School Students’ Mathematics Literacy Performance
    # Categorical features: Impute with most frequent, then one-hot encode.
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    # Ordinal features: Impute with most frequent (no scaling/encoding as their numerical order is meaningful).
    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])


    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),
                                                   ('cat', categorical_transformer, categorical_features),
                                                   ('ord', ordinal_transformer, ordinal_features)
                                                   ])

    return X_train, X_test, y_train, y_test, df_raw_with_correct_features, df_preprocessed, preprocessor

def create_imputed_dataframe(X_imputed: np.ndarray, preprocessor, y: pd.Series) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from the imputed numerical array, assigning correct column names
    from the preprocessor and re-attaching the target variable.

    Parameters:
        X_imputed (np.ndarray): The numpy array of imputed and transformed features.
        preprocessor (ColumnTransformer): The fitted ColumnTransformer used for transformation,
                                          which contains the `get_feature_names_out()` method.
        y (pd.Series): The target variable Series to be re-attached.

    Returns:
        pd.DataFrame: A DataFrame containing the imputed features and the target variable.
    """
    feature_names = preprocessor.get_feature_names_out()
    df = pd.DataFrame(X_imputed, columns=feature_names)
    df['PV1MATH'] = y.values
    return df

def create_preprocesser_without_drop_first(X_train: pd.DataFrame) -> ColumnTransformer:
    
    categorical_features = ['ST004D01T', 'SCHLTYPE']
    ordinal_features = ['ST062Q01TA', 'REPEAT', 'IMMIG'] 
    numerical_features = [col for col in X_train.columns if col not in categorical_features + ordinal_features] 

    numerical_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=8)),('yeo_johnson', PowerTransformer(method='yeo-johnson')),('scaler', StandardScaler())]) 
    
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(drop=None, handle_unknown='ignore'))])

    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),
                                                   ('cat', categorical_transformer, categorical_features),
                                                   ('ord', ordinal_transformer, ordinal_features)
                                                   ])
    return preprocessor

