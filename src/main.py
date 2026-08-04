import os
import shutil
import pandas as pd
import joblib

from src.config import MISSING_VALUES_THRESHOLD, CLASS_BOUNDARIES, CLASS_LABELS
from src.dataset_building import reduced_swiss_dataset
from src.dataset_cleaning_and_preprocessing import (
    create_eda_preprocessor,
    create_imputed_dataframe,
    run_preprocessing_pipeline,
)

from src.feature_selection import (
    run_feature_selection_pipeline,
)

from src.tables import (
    display_correlations_between_features,
    display_correlations_with_target,
    display_detailed_results_by_k,
    display_results,
    display_summary,
    generate_missing_values_report,
    get_descriptive_statistics,
)
from src.visualisations import (
    display_barplots,
    display_histograms,
    display_spearman_correlation_matrix,
    display_violin_plots,
    export_barplots,
    export_histograms,
    export_violin_plots,
)
from src.model_training import evaluate_k_values, run_models



if __name__ == "__main__":
    
    # 1. ========== LOAD / BUILD DATASET ==========
    df_raw = reduced_swiss_dataset()

    (X_train_imputed, X_test_imputed, X_train, X_test, y_train, y_test, df_raw_with_correct_features, df_preprocessed, numeric_imputed_feature_names, all_imputed_feature_names, removed_missing_columns) = run_preprocessing_pipeline(df_raw, MISSING_VALUES_THRESHOLD)

    output_dir = "outputs"
    dataframes_dir = os.path.join(output_dir, "dataframes")
    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")

    for dir_path in [output_dir, tables_dir, plots_dir, models_dir, dataframes_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 2. ========== GLOBAL EXPLORATORY DATA ANALYSIS ==========
    #get_descriptive_statistics(df_raw_with_correct_features, "GLOBAL DATA EXPLORATORY ANALYSIS (BEFORE CLEANING)")
    #get_descriptive_statistics(df_preprocessed, "GLOBAL DATA EXPLORATORY ANALYSIS (CLEANED WHOLE DATASET)")
    #print()
    #print("Columns removed from training-set missingness:", removed_missing_columns or "None")
    #print()
    #generate_missing_values_report(df_raw_with_correct_features, MISSING_VALUES_THRESHOLD, "MISSING VALUES REPORT (BEFORE CLEANING)")

    if os.path.exists("outputs/plots"):
        shutil.rmtree("outputs/plots")

    # Optional visualisations on original/interpretable units.
    #export_histograms(df_raw_with_correct_features, "BEFORE CLEANING")
    #export_violin_plots(df_raw_with_correct_features, "BEFORE CLEANING")
    #export_barplots(df_raw_with_correct_features, "BEFORE CLEANING")
    #display_histograms(df_raw_with_correct_features, "BEFORE CLEANING")
    #display_violin_plots(df_raw_with_correct_features, "BEFORE CLEANING")
    #display_barplots(df_raw_with_correct_features, "BEFORE CLEANING")


    # 3. ========== TRAIN-SET EDA AFTER IMPUTATION ==========
    eda_preprocessor = create_eda_preprocessor(X_train)
    X_train_imputed_for_eda = eda_preprocessor.fit_transform(X_train)
    X_test_imputed_for_eda = eda_preprocessor.transform(X_test)

    df_train_set_for_eda = create_imputed_dataframe(
        X_train_imputed_for_eda,
        eda_preprocessor,
        y_train,
    )

    #display_spearman_correlation_matrix(df_train_set_for_eda, "SPEARMAN CORRELATION MATRIX (TRAIN SET AFTER IMPUTATION)")
    #display_correlations_with_target(df_train_set_for_eda, "PV1MATH")
    #display_correlations_between_features(df_train_set_for_eda, "PV1MATH")

    # 4. ========== FEATURE SELECTION ==========

    #y_train/test for classification task
    y_train_class = pd.cut(y_train, bins = CLASS_BOUNDARIES, labels = CLASS_LABELS, right = False, include_lowest = True)
    y_test_class = pd.cut(y_test,bins = CLASS_BOUNDARIES, labels = CLASS_LABELS, right = False, include_lowest = True)

    # regression
    #(final_features_ranking, variance_treshold_df, ranking_MI_df, ranking_anova_df, ranking_rfe_df, X_train_after_var_thresh, X_test_after_var_thresh) = run_feature_selection_pipeline(X_train_imputed, X_test_imputed, y_train, all_imputed_feature_names, task = "regression")

    #classification
    (final_features_ranking, variance_treshold_df, ranking_MI_df, ranking_anova_df, ranking_rfe_df, X_train_after_var_thresh, X_test_after_var_thresh) = run_feature_selection_pipeline(X_train_imputed, X_test_imputed, y_train_class, all_imputed_feature_names, task = "classification")

    # 5. ========= MODEL TRAINING ===========
    print("\n" + "=" * 80)
    print("REGRESSION MODELS EVALUATION")
    print("=" * 80)

    
    # Evaluate regression models for different top-k features
    regression_results_df, regression_all_results_dic = evaluate_k_values(
        X_train_after_var_thresh, y_train, X_test_after_var_thresh, y_test,
        final_features_ranking = final_features_ranking,
        feature_names = all_imputed_feature_names,
        model_names=['LR', 'RF', 'XGBoost', 'SVR'],
        task='regression',
        k_values=[5, 10, 15, 20]
    )
    
    display_detailed_results_by_k(regression_all_results_dic)
    display_summary(regression_results_df, task='regression')

    # Save regression results DataFrame to tables folder
    regression_results_df.to_csv(os.path.join(dataframes_dir, 'regression_results_by_k.csv'), index=False)
    print(f" Regression results saved to '{os.path.join(dataframes_dir, 'regression_results_by_k.csv')}'")

    # Save regression all_results dictionary to models folder
    joblib.dump(regression_all_results_dic, os.path.join(models_dir, 'regression_all_results.pkl'))
    print(f" Regression all_results saved to '{os.path.join(models_dir, 'regression_all_results.pkl')}'")
    """

    print("\n" + "=" * 80)
    print("CLASSIFICATION MODELS EVALUATION")
    print("=" * 80)

    # Evaluate classification models for different top-k features
    classification_results_df, classification_all_results_dic = evaluate_k_values(
            X_train_after_var_thresh, y_train_class, X_test_after_var_thresh, y_test_class,
            final_features_ranking = final_features_ranking,
            feature_names = all_imputed_feature_names,
            model_names=['LR', 'RF', 'XGBoost', 'SVC'],
            task='classification',
            k_values=[5, 10, 15, 20]
        )

    display_detailed_results_by_k(classification_all_results_dic)
    display_summary(classification_results_df, task='classification')

    # Save classification results DataFrame to tables folder
    classification_results_df.to_csv(os.path.join(dataframes_dir, 'classification_results_by_k.csv'), index=False)
    print(f"Classification results saved to '{os.path.join(dataframes_dir, 'classification_results_by_k.csv')}'")

    # Save regression all_results dictionary to models folder
    joblib.dump(classification_all_results_dic, os.path.join(models_dir, 'classification_all_results.pkl'))
    print(f"Classification all_results saved to '{os.path.join(models_dir, 'classification_all_results.pkl')}'")
    """