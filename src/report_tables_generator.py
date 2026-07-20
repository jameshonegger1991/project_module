import pandas as pd
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_column

def generate_cleaning_summary_table(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame):
    """
    Generate a table that summarises the cleaning process.
    """
    summary_data = []
    
    # 1. Deleted columns (features)
    pct_columns_initial = get_missing_percentages_per_column(df_raw)
    
    print("=" * 120)
    print("PERCENTAGE OF MISSING VALUES PER FEATURE BEFORE CLEANING:")
    print("=" * 120)
    print()
    print(pct_columns_initial)
    
    deleted_cols = df_raw.columns.difference(df_cleaned.columns)

    if len(deleted_cols) == 0:
        summary_data.append({
            'Type': 'Columns',
            'Description': 'All features retained',
            'Action': 'NONE',
            'Detail': 'No feature removed (all thresholds OK)'
        })
    else:
        for col in deleted_cols:
            try:
                pct_val = float(pct_columns_initial[col])
            except (KeyError, ValueError):
                pct_val = 0.0
                
            summary_data.append({
                'Type': 'Column',
                'Description': col,
                'Action': 'DELETED',
                'Detail': f"{pct_val:.2f}% missing"
            })
    
    # 2. Deleted rows (observations)
    rows_deleted = len(df_raw) - len(df_cleaned)
    pct_rows_deleted = (rows_deleted / len(df_raw)) * 100 if len(df_raw) > 0 else 0
    
    summary_data.append({
        'Type': 'Rows',
        'Description': 'Total observations',
        'Action': f'REMOVED {rows_deleted}',
        'Detail': f'From {len(df_raw):,} to {len(df_cleaned):,} rows ({pct_rows_deleted:.1f}%)'
    })
    
    # 3. Summary statistics
    summary_data.append({
        'Type': 'SUMMARY',
        'Description': 'Final dataset',
        'Action': '',
        'Detail': f'{len(df_cleaned):,} rows x {len(df_cleaned.columns)} columns'
    })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Display with formatting
    print("\n" + "=" * 120)
    print("CLEANING DATASET SUMMARY")
    print("=" * 120)
    print(f"{'Type':<15} {'Description':<35} {'Action':<15} {'Detail'}")
    print("-" * 120)
    
    for item in summary_data:
        print(f"{item['Type']:<15} {item['Description']:<35} {item['Action']:<15} {item['Detail']}")
        
    print("=" * 120 + "\n")
    
    return summary_df
