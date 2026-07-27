import pandas as pd
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_column
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_row


def generate_missing_values_report(df: pd.DataFrame, threshold: float, title: str) -> None:
    """
    Generate a comprehensive missing values report for both columns and rows.
    
    Parameters:
        df: DataFrame to analyse
        threshold: Percentage threshold for highlighting high missingness (default: 70%)
    """
    col_pct = get_missing_percentages_per_column(df)
    row_pct = get_missing_percentages_per_row(df)

    #1. Columns with missing rate > threshold
    cols_above_threshold = col_pct[col_pct > threshold]
    cols_above_threshold_df = pd.DataFrame({'Column': cols_above_threshold.index,'Missing %': cols_above_threshold.values}).sort_values(by = 'Missing %', ascending = False)

    #2. Rows statistics
    total_rows = len(df)
    rows_missing_50 = (row_pct >= 50).sum()
    rows_missing_70 = (row_pct >= 70).sum()
    rows_missing_90 = (row_pct >= 90).sum()
    rows_above_threshold = row_pct[row_pct > threshold]

    rows_with_missing = row_pct[row_pct > 0]  
    mean_missing_pct = rows_with_missing.mean() if len(rows_with_missing) > 0 else 0
    median_missing_pct = rows_with_missing.median() if len(rows_with_missing) > 0 else 0
    rows_with_missing_count = len(rows_with_missing)

    #3. Display
    
    print()
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    print()
    print("\n=== COLUMNS ===")
    print(f"Total columns: {len(col_pct)}")
    print(f"Columns with > {threshold}% missing: {len(cols_above_threshold)}")
    if len(cols_above_threshold) > 0:
        print(f"\nColumns with missingness above {threshold}%:")
        print(cols_above_threshold_df.to_string(index=False))
    
    print("\n=== ROWS ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with > {threshold}% missing: {len(rows_above_threshold):,} ({len(rows_above_threshold)/total_rows*100:.1f}%)")
    print(f"Rows with ≥ 50% missing: {rows_missing_50:,} ({rows_missing_50/total_rows*100:.1f}%)")
    print(f"Rows with ≥ 70% missing: {rows_missing_70:,} ({rows_missing_70/total_rows*100:.1f}%)")
    print(f"Rows with ≥ 90% missing: {rows_missing_90:,} ({rows_missing_90/total_rows*100:.1f}%)")
    
    print(f"\n=== Rows with missing values ===")
    print(f"Rows with at least one missing value: {rows_with_missing_count:,} ({rows_with_missing_count/total_rows*100:.1f}%)")
    if rows_with_missing_count > 0:
        print(f"Mean missing % per row (among rows with missing): {mean_missing_pct:.1f}%")
        print(f"Median missing % per row (among rows with missing): {median_missing_pct:.1f}%")
        print(f"Average missing cells per row: {(rows_with_missing / 100 * len(df.columns)).mean():.1f} cells")


    print("\n=== SUMMARY ===")
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    print(f"Total cells: {total_cells:,}")
    print(f"Total missing cells: {total_missing:,}")
    print(f"Overall missing %: {(total_missing / total_cells) * 100:.1f}%")
    print("=" * 80 + "\n")

def get_descriptive_statistics(df: pd.DataFrame, title: str) -> None:
    """
    Display descriptive statistics for a DataFrame.
    It includes shape, descriptive stats, head, info, and skewness.
    """
    
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()
    
    # 1. Shape
    print("=" * 80)
    print("SHAPE OF THE DATASET")
    print("=" * 80)
    print()
    print(df.shape)
    print()

    # 2. Head
    print("=" * 80)
    print("HEAD OF THE DATASET")
    print("=" * 80)
    print()
    print(df.head())
    print()
    
    # 3. Descriptive statistics
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    print()
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print(numeric_df.describe())
    print()
    
    # 4. Info
    print("=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print()
    type_summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    print(type_summary.to_string(index=False))
    print()
    
    # 5. Skewness
    print("=" * 80)
    print("SKEWNESS OF THE DATASET")
    print("=" * 80)
    print()
    print(numeric_df.skew())
    print()
    
    print("=" * 80 + "\n")

def display_correlations_with_target(df: pd.DataFrame, target: str = "PV1MATH"):
    """
    Displays all variables according to their Spearman correlation
    with the target variable.

    Positive and negative correlations are displayed separately.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot calculate correlations.")
        return

    if target not in df.columns:
        print(f"The target variable '{target}' was not found.")
        return

    # Calculate correlations with the target and remove the target itself.
    correlations = (df.corr(method="spearman")[target].drop(target))

    # Sort positive correlations from strongest to weakest.
    positive_correlations = (correlations[correlations > 0].sort_values(ascending=False).rename("Correlation").to_frame())

    # Sort negative correlations from strongest to weakest.
    positive_correlations.index.name = "Feature"

    negative_correlations = (correlations[correlations < 0].sort_values(ascending=True).rename("Correlation").to_frame())

    negative_correlations.index.name = "Feature"

    print(f"\nPOSITIVE CORRELATIONS WITH {target}")
    print(positive_correlations)

    print(f"\nNEGATIVE CORRELATIONS WITH {target}")
    print(negative_correlations)

def display_correlations_between_features(df: pd.DataFrame, target: str = "PV1MATH", corr_threshold: float = 0.70):
    """
    Displays the 20 strongest positive and negative Spearman correlations
    between features, excluding the target variable.

    It also displays feature pairs whose absolute correlation exceeds
    the specified threshold, indicating a potential risk of multicollinearity.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target (str): The target variable to exclude.
    - corr_threshold (float): The correlation threshold used to identify
      a potential risk of multicollinearity. By default, it is set to 0.70, 
      representing a relatively conservative threshold.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot calculate correlations.")
        return

    # Remove the target variable before calculating correlations.
    features = df.drop(columns=[target], errors="ignore")
    correlation_matrix = features.corr(method="spearman")
    correlations = []

    # It keeps each variable pair only once.
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            correlations.append({
                "Feature 1": correlation_matrix.columns[i],
                "Feature 2": correlation_matrix.columns[j],
                "Correlation": correlation_matrix.iloc[i, j]
            })

    correlations_df = pd.DataFrame(correlations).dropna()

    # Selection of  the 20 strongest positive correlations.
    positive_correlations = (
        correlations_df[correlations_df["Correlation"] > 0]
        .sort_values("Correlation", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # Selection of the 20 strongest negative correlations.
    negative_correlations = (
        correlations_df[correlations_df["Correlation"] < 0]
        .sort_values("Correlation", ascending=True)
        .head(20)
        .reset_index(drop=True)
    )

    # Select correlations exceeding the multicollinearity threshold.
    # The literature recommends a threshold between 0.7 - 0.8 
    # SOURCE:
    # - Arroyo Resino et al. - 2024 - Student well-being and mathematical literacy performance in PISA 2018 a machine-learning approach.
    # - Masci et al. - 2018 - Student and school performance across countries A machine learning approach.
    multicollinearity_risk = (
        correlations_df[
            correlations_df["Correlation"].abs() >= corr_threshold
        ]
        .assign( # Using .assign() avoids a SettingWithCopyWarning by returning a new DataFrame.
            Absolute_Correlation = lambda x: x["Correlation"].abs() # INSPIRATION: https://medium.com/@whyamit404/understanding-pandas-assign-with-step-by-step-examples-6cf2d79ff481 
        )
        .sort_values("Absolute_Correlation", ascending=False)
        .reset_index(drop=True)
    )

    print("\n20 STRONGEST POSITIVE CORRELATIONS BETWEEN FEATURES")
    print(positive_correlations)

    print("\n20 STRONGEST NEGATIVE CORRELATIONS BETWEEN FEATURES")
    print(negative_correlations)

    print(
        f"\nPOTENTIAL MULTICOLLINEARITY RISK "
        f"(|CORRELATION| >= {corr_threshold})"
    )

    if multicollinearity_risk.empty:
        print("No feature pair exceeds the selected corr_threshold.")
    else:
        print(multicollinearity_risk)

def variance_threshold_report(threshold: float, numeric_cols: list[str], selected_features: list[str]) -> None:
    """
    Display numerical features retained and removed by VarianceThreshold.
    """
    
    removed_features = [col for col in numeric_cols if col not in selected_features]
    print(f"Variance threshold ({threshold}) applied to numerical columns.")
    print(f" Columns kept : {len(selected_features)}/{len(numeric_cols)}")
    if removed_features:
        print(f" Columns deleted : {removed_features}")