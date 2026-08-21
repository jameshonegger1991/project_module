import os
import pandas as pd

from src.config import (
    MISSING_VALUES_THRESHOLD, 
    CLASS_BOUNDARIES, 
    CLASS_LABELS,
    OUTPUTS_DIR,
    TABLES_DIR,
    PLOTS_DIR,
)
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
    save_correlations_among_features,
    save_correlations_with_target,
    save_detailed_results_by_k,
    save_summary_of_metrics,
    save_missing_values_report,
    save_descriptive_statistics,
)

from src.visualisations import (
    barplot_global_shap_rankings,
    save_combined_barplots,
    save_combined_histograms,
    save_spearman_correlation_matrix,
    save_combined_violin_plots,
    save_individual_barplots,
    save_individual_histograms,
    save_individual_violin_plots,
    save_confusion_matrices_plot_all_models,
    save_learning_curves_plot_all_models,
    save_multiple_metrics_vs_features_plot,
    save_residuals_plot_from_all_results,
    shap_summary_plot,
)
from src.model_training import evaluate_k_values, run_models
from src.SHAP_analysis import (
    run_complete_shap_analysis_and_classification,
)

from src.utils import (
    print_files,
    display_visualisation,
)


if __name__ == "__main__":

    # =========================================================
    # ========== I. BUILD AND LOAD THE DATASET ================
    # =========================================================
    
    print("\n" + "*" * 100)
    print("I. BUILD AND LOAD THE DATASET")
    print("*" * 100)
    
    df_raw = "dataset/swiss_reduced_dataset.csv"

    if not os.path.exists(df_raw):
        # build the reduced swiss dataset from PISA 2022 dataset (requires manual download)
        df_raw = reduced_swiss_dataset()
    else:
        df_raw = pd.read_csv(df_raw)
    print()

    # =========================================================
    # ============== II. PREPROCESSING  =======================
    # =========================================================
    
    print("*" * 100)
    print("II. PREPROCESSING PROCESS")
    print("*" * 100)
    
    (X_train_imputed, 
     X_test_imputed, 
     X_train, 
     X_test, 
     y_train, 
     y_test, 
     df_raw_with_correct_features, 
     df_preprocessed, 
     numeric_imputed_feature_names, 
     all_imputed_feature_names, 
     removed_missing_columns) = run_preprocessing_pipeline(df_raw, MISSING_VALUES_THRESHOLD)
    print()

    # =========================================================
    # ========== III. GLOBAL EXPLORATORY DATA ANALYSIS ========
    # =========================================================
    
    print("*" * 100)
    print("III. GLOBAL EXPLORATORY DATA ANALYSIS")
    print("*" * 100)
    
    for dir_path in [OUTPUTS_DIR, TABLES_DIR, PLOTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Data analysis: generate tables
    save_descriptive_statistics(df_raw_with_correct_features, "Global Data Exploratory Analysis (whole dataset before cleaning)")
    save_descriptive_statistics(df_preprocessed, "Global Data Exploratory Analysis (Cleaned Whole Dataset)")
    save_missing_values_report(df_raw_with_correct_features, MISSING_VALUES_THRESHOLD, "Missing Values Report (before cleaning)")

    # Data analysis: generate plots 
    save_individual_histograms(df_raw_with_correct_features, "Before Cleaning")
    save_individual_violin_plots(df_raw_with_correct_features, "Before Cleaning")
    save_individual_barplots(df_raw_with_correct_features, "Before Cleaning")
    save_combined_histograms(df_raw_with_correct_features)
    save_combined_violin_plots(df_raw_with_correct_features)
    save_combined_barplots(df_raw_with_correct_features)

    # Data analysis: display tables and plots
    print_files("outputs/tables/EDA/whole_dataset/Global Data Exploratory Analysis (whole dataset before cleaning).txt")
    print_files("outputs/tables/EDA/whole_dataset/Missing Values Report (before cleaning).txt")
    display_visualisation("outputs/plots/EDA/whole_dataset/univariate_plots/barplots/Bar Plots - All Categorical Features (Before Cleaning And Imputation).png")
    display_visualisation("outputs/plots/EDA/whole_dataset/univariate_plots/histograms/Histograms - All Numeric Features (Before Cleaning And Imputation).png")
    display_visualisation("outputs/plots/EDA/whole_dataset/univariate_plots/violin_plots/Violin Plots - All Numeric Features (Before Cleaning And Imputation).png")
    print_files("outputs/tables/EDA/whole_dataset/Global Data Exploratory Analysis (Cleaned Whole Dataset).txt")

    # ===========================================================
    # IV. EXPLORATORY DATA ANALYSIS ON TRAIN-SET AFTER IMPUTATION 
    # ===========================================================
    
    print("*" * 100)
    print("IV. EXPLORATORY DATA ANALYSIS ON TRAIN-SET AFTER IMPUTATION")
    print("*" * 100)

    eda_preprocessor = create_eda_preprocessor(X_train)
    X_train_imputed_for_eda = eda_preprocessor.fit_transform(X_train)

    df_train_set_for_eda = create_imputed_dataframe(
    X_train_imputed_for_eda,
    eda_preprocessor,
    y_train,
    )

    # Data analysis: generate and save correlation computations
    save_spearman_correlation_matrix(df_train_set_for_eda, "Spearman Correlation Matrix (Train set after imputation)")
    save_correlations_with_target(df_train_set_for_eda, "PV1MATH")
    save_correlations_among_features(df_train_set_for_eda, "PV1MATH")

    # Data analysis: display correlation tables and correlation matrix
    display_visualisation("outputs/plots/EDA/train_set/multivariate_plots/Spearman Correlation Matrix (Train set after imputation).png")
    print_files("outputs/tables/EDA/train_set/correlations_with_PV1MATH.txt")
    print_files("outputs/tables/EDA/train_set/correlations_among_features.txt")

    
    # ==============================================================
    # V. FEATURE SELECTION (FOR CLASSIFICATION AND REGRESSION TASKS) 
    # ==============================================================
    
    print("*" * 100)
    print("V. FEATURE SELECTION (FOR CLASSIFICATION AND REGRESSION TASKS)")
    print("*" * 100)

    #y_train/test for classification task
    y_train_class = pd.cut(y_train, bins = CLASS_BOUNDARIES, labels = CLASS_LABELS, right = False, include_lowest = True)
    y_test_class = pd.cut(y_test,bins = CLASS_BOUNDARIES, labels = CLASS_LABELS, right = False, include_lowest = True)

    # regression
    (final_features_ranking_reg, 
     variance_threshold_df_reg, 
     ranking_MI_df_reg, 
     ranking_anova_df_reg, 
     ranking_rfe_df_reg, 
     X_train_after_var_thresh_reg, 
     X_test_after_var_thresh_reg, 
     selected_columns_var_thresh_reg) = run_feature_selection_pipeline(X_train_imputed, 
                                                                       X_test_imputed, y_train, 
                                                                       all_imputed_feature_names, 
                                                                       task = "regression")
    
    #classification
    (final_features_ranking_class, 
     variance_threshold_df_class, 
     ranking_MI_df_class, 
     ranking_anova_df_class, 
     ranking_rfe_df_class, 
     X_train_after_var_thresh_class, 
     X_test_after_var_thresh_class, 
     selected_columns_var_thresh_class) = run_feature_selection_pipeline(X_train_imputed, 
                                                                         X_test_imputed, 
                                                                         y_train_class, 
                                                                         all_imputed_feature_names, 
                                                                         task = "classification")

    print_files("outputs/tables/feature_selection/variance_threshold_ranking.txt")

    print(f"***** REGRESSION TASK *****")
    print_files("outputs/tables/feature_selection/regression_mutual_info_ranking.txt")
    print_files("outputs/tables/feature_selection/regression_anova_ranking.txt")
    print_files("outputs/tables/feature_selection/regression_rfe_ranking.txt")
    print_files("outputs/tables/feature_selection/regression_combined_feature_selection_rankings.txt")

    print(f"***** CLASSIFICATION TASK *****")
    print_files("outputs/tables/feature_selection/classification_mutual_info_ranking.txt")
    print_files("outputs/tables/feature_selection/classification_anova_ranking.txt")
    print_files("outputs/tables/feature_selection/classification_rfe_ranking.txt")
    print_files("outputs/tables/feature_selection/classification_combined_feature_selection_rankings.txt")

    # ==============================================================
    # ========== VI.I MODEL TRAINING FOR REGRESSION TASK  ==========
    # ==============================================================
    
    print("*" * 100)
    print("VI.I MODEL TRAINING FOR REGRESSION TASK")
    print("*" * 100)

    regression_results_df, regression_all_results_dic = evaluate_k_values(
        X_train_after_var_thresh_reg,
        y_train,
        X_test_after_var_thresh_reg,
        y_test,
        final_features_ranking=final_features_ranking_reg,
        feature_names=selected_columns_var_thresh_reg,
        model_names=['LR', 'RF', 'XGBoost', 'SVR'],
        task='regression',
        k_values=[5, 10, 15, 20]
    )

    save_detailed_results_by_k(regression_all_results_dic, task='regression')
    save_summary_of_metrics(regression_results_df, task='regression')
    save_multiple_metrics_vs_features_plot(regression_results_df, task='regression')

    k_chosen = 10
    save_residuals_plot_from_all_results(regression_all_results_dic, k_chosen=k_chosen)
    save_learning_curves_plot_all_models(regression_all_results_dic, k_chosen=k_chosen, X_train_after_var_thresh=X_train_after_var_thresh_reg, y_train=y_train, task='regression')

    # Display model training results, tables and plots
    print_files(f"{TABLES_DIR}/model_training/regression_detailed_results_by_k.txt")
    print_files(f"{TABLES_DIR}/model_training/regression_summary_of_metrics.txt")

    display_visualisation(f"{PLOTS_DIR}/model_training/regression_metrics_vs_features_plot.png")
    display_visualisation(f"{PLOTS_DIR}/model_training/regression_residual_plots_for_{k_chosen}_features.png")
    display_visualisation(f"{PLOTS_DIR}/model_training/regression_learning_curves_top_{k_chosen}_features.png")


    # ==============================================================
    # ======= VI.II MODEL TRAINING FOR CLASSIFICATION TASK  ========
    # ==============================================================
    
    print("*" * 100)
    print("VI.II MODEL TRAINING FOR CLASSIFICATION TASK")
    print("*" * 100)
    
    # Evaluate classification models for different top-k features
    classification_results_df, classification_all_results_dic = evaluate_k_values(
        X_train_after_var_thresh_class, y_train_class, X_test_after_var_thresh_class, y_test_class,
        final_features_ranking = final_features_ranking_class,
        feature_names = selected_columns_var_thresh_class,
        model_names=['LR', 'RF', 'XGBoost', 'SVC'],
        task='classification',
        k_values=[5, 10, 15, 20]
        )

    save_detailed_results_by_k(classification_all_results_dic, task='classification')
    save_summary_of_metrics(classification_results_df, task='classification')

    save_multiple_metrics_vs_features_plot(classification_results_df, task='classification')
    save_confusion_matrices_plot_all_models(classification_all_results_dic, k_chosen=10, class_labels=CLASS_LABELS)

    save_learning_curves_plot_all_models(
        classification_all_results_dic, 
        k_chosen=10, 
        X_train_after_var_thresh=X_train_after_var_thresh_class, 
        y_train=y_train_class, 
        task='classification'
    )

    k_chosen = 10
    # Display model training results, tables and plots
    print_files(f"{TABLES_DIR}/model_training/classification_detailed_results_by_k.txt")
    print_files(f"{TABLES_DIR}/model_training/classification_summary_of_metrics.txt")

    display_visualisation(f"{PLOTS_DIR}/model_training/classification_metrics_vs_features_plot.png")
    display_visualisation(f"{PLOTS_DIR}/model_training/classification_learning_curves_top_{k_chosen}_features.png")
    display_visualisation(f"{PLOTS_DIR}/model_training/confusion_matrices_top_{k_chosen}_features.png")

    # ==============================================================
    # ================= VII. SHAP ANALYSIS =========================
    # ==============================================================
    
    print("*" * 100)
    print("VII. SHAP ANALYSIS")
    print("*" * 100)

    (global_shap_rankings, 
    shap_values_for_all_models, 
    X_test_k, 
    feature_names_k, 
    number_of_features_chosen_for_model_training) = run_complete_shap_analysis_and_classification(
        X_train = X_train_after_var_thresh_reg,
        X_test = X_test_after_var_thresh_reg,
        number_of_features_chosen_for_model_training = 10,
        task ='regression',
        all_training_model_results_dic = regression_all_results_dic,
        rashomon_set_threshold = 0.05,
        number_of_features_retained_for_final_classification = 5,
        )

    print_files("outputs/tables/SHAP_analysis/rashomon_set.txt")
    print_files("outputs/tables/SHAP_analysis/global_shap_rankings.txt")

    barplot_global_shap_rankings(global_shap_rankings, number_of_features_chosen_for_model_training)
    shap_summary_plot(shap_values_for_all_models, feature_names_k)

    display_visualisation(f"{PLOTS_DIR}/SHAP_analysis/Global_SHAP_feature_importances_barplot.png")
    display_visualisation(f"{PLOTS_DIR}/SHAP_analysis/SHAP_summary_plot.png")

    print_files("outputs/tables/SHAP_analysis/feature_agreement_stats.txt")
    print_files("outputs/tables/SHAP_analysis/inter_model_concordance_agreement.txt")
    print_files("outputs/tables/SHAP_analysis/intra_model_stability_assessment.txt")
    print_files("outputs/tables/SHAP_analysis/feature_robustness_assessment.txt")
        