from src.dataset_building import build_swiss_merged_dataset
from src.dataset_building import reduced_swiss_dataset
from src.dataset_cleaning_and_preprocessing import add_MEAN_ESCS_and_MMINS_features
from src.dataset_cleaning_and_preprocessing import missing_values_removal
from src.dataset_cleaning_and_preprocessing import data_cleaning_preprocessing_pipeline
from src.report_tables_generator import generate_cleaning_summary_table


if __name__ == "__main__":
    
    #1. Load (or build) the dataset
    df_raw = reduced_swiss_dataset()
    
    #2. Clean and preprocess the dataset (features mapping, encoding, train/split)
    X_train, X_test, y_train, y_test, df_with_missing_values, df_preprocessed, preprocessor = data_cleaning_preprocessing_pipeline(df_raw)
    
    #3. Generate statistics table about the missing values
    generate_cleaning_summary_table(df_raw, df_preprocessed)

