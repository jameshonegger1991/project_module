from src.dataset_building import reduced_swiss_dataset
from src.dataset_cleaning_and_preprocessing import data_preprocessing_pipeline
from src.tables import generate_missing_values_report
from src.dataset_cleaning_and_preprocessing import create_imputed_dataframe, create_preprocesser_without_drop_first
from src.tables import get_descriptive_statistics, display_correlations_with_target, display_correlations_between_features
from src.visualisations import export_barplots, export_histograms, export_violin_plots, display_histograms, display_violin_plots, display_barplots, display_spearman_correlation_matrix
import pandas as pd
import os 
import shutil

if __name__ == "__main__":
  
    
    #1. ========== LOAD/BUILD DATASET ==========
    df_raw = reduced_swiss_dataset()
    X_train, X_test, y_train, y_test, df_raw_with_correct_features, df_preprocessed, preprocessor = data_preprocessing_pipeline(df_raw, 70)
    

    #2. ========== GLOBAL DATA EXPLORATORY ANALYSIS (WHOLE DATASET) ==========
    #get_descriptive_statistics(df_preprocessed, "GLOBAL DATA EXPLORATORY ANALYSIS (WHOLE DATASET AFTER PREPROCESSING)")

    #get_descriptive_statistics(df_raw_with_correct_features, "GLOBAL DATA EXPLORATORY ANALYSIS (WHOLE DATASET BEFORE PREPROCESSING)")
    #generate_missing_values_report(df_raw_with_correct_features, 70, "MISSING VALUES REPORT (WHOLE DATASET BEFORE PREPROCESSING)")

    #delete plots folder and subfolders from previous run.
    if os.path.exists("plots"):
        shutil.rmtree("plots")

    #export_histograms(df_raw_with_correct_features, "HISTOGRAM (WHOLE DATASET BEFORE PREPROCESSING)")
    #export_violin_plots(df_raw_with_correct_features, "VIOLIN PLOT (WHOLE DATASET BEFORE PREPROCESSING)")
    #export_barplots(df_raw_with_correct_features, "BAR PLOT (WHOLE DATASET BEFORE PREPROCESSING)")

    #display_histograms(df_raw_with_correct_features, "HISTOGRAMS FOR ALL NUMERIC FEATURES (WHOLE DATASET BEFORE PREPROCESSING)")
    #display_violin_plots(df_raw_with_correct_features, "VIOLIN PLOTS FOR ALL NUMERIC FEATURES (WHOLE DATASET BEFORE PREPROCESSING)")
    #display_barplots(df_raw_with_correct_features, "BAR PLOTS FOR ALL CATEGORICAL FEATURES (WHOLE DATASET BEFORE PREPROCESSING)")
    

    #3. ========== MONITORED (POST IMPUTATION) DATA EXPLORATORY ANALYSIS (TRAIN SET) ==========
    preprocessor_without_drop_first = create_preprocesser_without_drop_first(X_train)
    X_train_imputed_for_EDA = preprocessor_without_drop_first.fit_transform(X_train)
    X_test_imputed_for_EDA = preprocessor_without_drop_first.transform(X_test)
    df_train_set_for_EDA = create_imputed_dataframe(X_train_imputed_for_EDA, preprocessor_without_drop_first, y_train)
    #get_descriptive_statistics(df_train_set_for_EDA, "MONITORED DATA EXPLORATORY ANALYSIS (TRAIN SET AFTER PREPROCESSING)")
    #generate_missing_values_report(df_train_set_for_EDA, 70.0, "MISSING VALUES REPORT (TRAIN SET AFTER PREPROCESSING)")
    #export_histograms(df_train_set_for_EDA, "HISTOGRAM (TRAIN SET AFTER PREPROCESSING)")
    #export_violin_plots(df_train_set_for_EDA, "VIOLIN PLOT (TRAIN SET AFTER PREPROCESSING)")

    #display_violin_plots(df_train_set_for_EDA, "VIOLIN PLOTS FOR ALL NUMERIC FEATURES (TRAIN SET AFTER PREPROCESSING)")

    #display_histograms(df_train_set_for_EDA, "HISTOGRAMS FOR ALL NUMERIC FEATURES (TRAIN SET AFTER PREPROCESSING)")
    #display_spearman_correlation_matrix(df_train_set_for_EDA, "SPEARMAN CORRELATION MATRIX (TRAIN SET AFTER PREPROCESSING)")
    display_correlations_with_target(df_train_set_for_EDA, "PV1MATH")
    display_correlations_between_features(df_train_set_for_EDA, "PV1MATH")

                     

    #4. ========== FEATURE SELECTION ==========
    

