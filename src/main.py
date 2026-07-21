from src.dataset_building import reduced_swiss_dataset
from src.dataset_cleaning_and_preprocessing import data_preprocessing_pipeline
from src.tables import generate_missing_values_report
from src.dataset_cleaning_and_preprocessing import create_imputed_dataframe
from src.tables import get_descriptive_statistics
import pandas as pd


if __name__ == "__main__":
    
    #1. ========== LOAD/BUILD DATASET ==========
    df_raw = reduced_swiss_dataset()
    X_train, X_test, y_train, y_test, df_raw_with_correct_features, df_preprocessed, preprocessor = data_preprocessing_pipeline(df_raw, 70)
    

    #2. ========== GLOBAL DATA EXPLORATORY ANALYSIS (WHOLE DATASET) ==========
    get_descriptive_statistics(df_raw_with_correct_features, "GLOBAL DATA EXPLORATORY ANALYSIS (WHOLE DATASET BEFORE PREPROCESSING)")
    generate_missing_values_report(df_raw_with_correct_features, 70, "MISSING VALUES REPORT (WHOLE DATASET BEFORE PREPROCESSING)")

    

    #3. ========== MONITORED (POST IMPUTATION) DATA EXPLORATORY ANALYSIS (TRAIN SET) ==========
    X_train_imputed = preprocessor.fit_transform(X_train)
    X_test_imputed = preprocessor.transform(X_test)
    df_train_set = create_imputed_dataframe(X_train_imputed, preprocessor, y_train)
    generate_missing_values_report(df_train_set, 70.0, "MISSING VALUES REPORT (TRAIN SET AFTER PREPROCESSING)")


    #4. ========== FEATURE SELECTION ==========
  

