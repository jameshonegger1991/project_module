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

