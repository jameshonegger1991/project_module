from src.dataset_building import build_swiss_merged_dataset
from src.dataset_building import reduced_swiss_dataset
from src.dataset_cleaning_and_preprocessing import add_MEAN_ESCS_and_MMINS_features
from src.dataset_cleaning_and_preprocessing import missing_values_removal
from src.report_tables_generator import generate_cleaning_summary_table


if __name__ == "__main__":
    df = add_MEAN_ESCS_and_MMINS_features()
    #print(df.head()) 
    #print(df.shape) 
    #print(list(df.columns))
    
    df_cleaned = missing_values_removal(df)
    #print(df_cleaned.head())
    #print(df_cleaned.shape)
    generate_cleaning_summary_table(df, df_cleaned)
    

