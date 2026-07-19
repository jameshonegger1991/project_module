import pandas as pd
import os


def add_MEAN_ESCS_and_MMINS_features()-> pd.DataFrame :
    
    dataset_path = os.path.join("src", "swiss_reduced_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        print("First, please run reduced_swiss_dataset().")
        return None
    
    df = pd.read_csv(dataset_path)
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


def missing_values_removal(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns and rows with more than 70% missing values."""
    # 1. Delete columns with missing values > 70% (based on Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024))
    pct_columns = get_missing_percentages_per_column(df)
    columns_to_keep = pct_columns[pct_columns <= 70].index
    df_step1 = df[columns_to_keep]

    # 2. Delete lines with missing values > 70% (based on Gamazo & Martínez-Abad (2020), Huang, Y., Zhou, Y., Chen, J., & Wu, D. (2024))
    pct_rows = get_missing_percentages_per_row(df_step1)
    df_step2 = df_step1[pct_rows <= 70]

    # 3. Delete lines with missing value in target variable (PV1MATH)
    df_cleaned = df_step2.dropna(subset=['PV1MATH'])

    return df_cleaned
    
    
#NORMALISATION, SCALING ET TOUT çA
