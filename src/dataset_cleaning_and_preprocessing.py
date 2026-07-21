import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def add_MEAN_ESCS_and_MMINS_features(df: pd.DataFrame)-> pd.DataFrame :
    """ 
    This function adds two new features to the dataset: MEAN_ESCS and MMINS. 
    It also removes identifier features (CNTSTUID and CNTSCHID) and the features used to build MEAN_ESCS and MMINS,
    to reduce bias and multicolinearity during training.
    """
    df['MMINS'] = df['ST059Q01TA'] * df['SC175Q01JA']
    df['MEAN_ESCS'] = df.groupby('CNTSCHID')['ESCS'].transform('mean')

    #Original features used to build MEAN_ESCS/MMINS indexes and non-significative features are removed. 
    df = df.drop(columns=['ESCS', 'ST059Q01TA', 'SC175Q01JA', 'CNTSCHID','CNTSTUID'])
    
    return df


def get_missing_percentages_per_column(df):
    """Compute the percentage of missing values for each column in a DataFrame."""
    return (df.isnull().sum() / len(df)) * 100

def get_missing_percentages_per_row(df):
    """Compute the percentage of missing values for each row in a DataFrame."""
    return (df.isnull().sum(axis=1) / len(df.columns)) * 100

def missing_values_removal(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Remove columns and rows with more than 70% missing values."""
    # 1. Delete columns with missing values >= 70% (based on Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024))
    pct_columns = get_missing_percentages_per_column(df)
    columns_to_keep = pct_columns[pct_columns < threshold].index
    df_step1 = df[columns_to_keep]

    # 2. Delete lines with missing values >= 70% (based on Gamazo & Martínez-Abad (2020), Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024))
    pct_rows = get_missing_percentages_per_row(df_step1)
    df_step2 = df_step1[pct_rows < threshold]

    # 3. Delete lines with missing value in target variable (PV1MATH)
    df_cleaned = df_step2.dropna(subset=['PV1MATH'])

    return df_cleaned

def ordinal_features_mapping(df: pd.DataFrame):
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

def data_preprocessing_pipeline(df: pd.DataFrame, threshold: float):
    """
    Data preprocessing pipeline following strict data leakage prevention.
       
    1. Train/Test Split:
       - The filtered dataset is split into train (80%) and test (20%) sets.
       
    3. Transformation Pipeline (on the TRAINING set):
       - Imputation: Remaining missing values are imputed using the median for numeric features, 
         and the mode for categorical features.
       - Nominal features (ST004D01T, SCHLTYPE) are one-hot encoded with drop='first' 
         to avoid multicollinearity.
       - Ordinal (ST062Q01TA, IMMIG, REPEAT) are preserved as numeric to maintain 
         their hierarchical order.
       - All numeric features are standardised using Z-score scaling (StandardScaler), a process strictly required for models like MLR, SVR, and KNN.
    """
    #1. Initial Filtering
    df_raw_with_correct_features = add_MEAN_ESCS_and_MMINS_features(df)
    df_mapped = ordinal_features_mapping(df_raw_with_correct_features.copy())
    df_preprocessed = missing_values_removal(df_mapped, threshold)

    #2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop(columns = ['PV1MATH']), df_preprocessed['PV1MATH'], test_size=0.2, random_state = 7) # Following common practice in the literature, a 20% test set is used.

    #3. Transformation Pipeline
    categorical_features = ['ST004D01T', 'SCHLTYPE'] 
    ordinal_features = ['ST062Q01TA', 'REPEAT', 'IMMIG'] 
    numerical_features = [col for col in X_train.columns if col not in categorical_features + ordinal_features] 

    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])
    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),
                                                   ('cat', categorical_transformer, categorical_features),
                                                   ('ord', ordinal_transformer, ordinal_features)
                                                   ])
    
    return X_train, X_test, y_train, y_test, df_raw_with_correct_features, df_preprocessed, preprocessor

def create_imputed_dataframe(X_imputed: np.ndarray, preprocessor, y: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame from imputed data with proper column names.
    """
    feature_names = preprocessor.get_feature_names_out()
    df = pd.DataFrame(X_imputed, columns=feature_names)
    df['PV1MATH'] = y.values
    return df