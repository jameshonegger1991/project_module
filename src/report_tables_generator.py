import pandas as pd
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_column

def generate_cleaning_summary_table(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame):
    """
    Generate a table that summarises the cleaning process.
    """
    summary_data = []
    
    # 1. Deleted columns (features)
    pct_columns_initial = get_missing_percentages_per_column(df_raw)
    deleted_cols = df_raw.columns.difference(df_cleaned.columns)
    
    for col in deleted_cols:
        summary_data.append({
            'Type': 'Column',
            'Name': col,
            'Action': 'DELETED',
            'Detail': f"{pct_columns_initial.get(col, 0):.2f}% missing"
        })
    
    # 2. Deleted rows (observations)
    rows_deleted = len(df_raw) - len(df_cleaned)
    pct_rows_deleted = (rows_deleted / len(df_raw)) * 100
    
    summary_data.append({
        'Type': 'Rows',
        'Name': 'Total observations',
        'Action': f'REMOVED {rows_deleted}',
        'Detail': f'From {len(df_raw):,} to {len(df_cleaned):,} rows ({pct_rows_deleted:.1f}%)'
    })
    
    # 3. Summary statistics
    summary_data.append({
        'Type': 'SUMMARY',
        'Name': 'Final dataset',
        'Action': 'READY',
        'Detail': f'{len(df_cleaned):,} rows x {len(df_cleaned.columns)} columns'
    })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Display with formatting
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"{'Type':<15} {'Name':<35} {'Action':<15} {'Detail'}")
    print("-"*80)
    for item in summary_data:
        print(f"{item['Type']:<15} {item['Name']:<35} {item['Action']:<15} {item['Detail']}")
    print("="*60 + "\n")
    
    return summary_df