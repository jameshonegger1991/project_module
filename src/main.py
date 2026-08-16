import os
import shutil
import pandas as pd
import joblib

from src.config import (
    MISSING_VALUES_THRESHOLD, 
    CLASS_BOUNDARIES, 
    CLASS_LABELS,
    OUTPUTS_DIR,
    TABLES_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    SAVEDFILES_DIR,
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
    save_rashomon_set,
    save_results,
    save_summary_of_metrics,
    save_global_shap_rankings,
    save_missing_values_report,
    save_descriptive_statistics,
    save_inter_model_concordance_agreement,
    save_feature_agreement_stats,
    save_feature_robustness_assessment,
    save_intra_model_stability_assessment,
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
    local_and_global_shap_values_calculator, 
    rashomon_set_builder, 
    inter_model_concordance_assessment, 
    feature_agreement_stats,
    intra_model_stability_assessment,
    assess_features_robustness,
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
    
    print("\n" + "=" * 80)
    print("I. BUILD AND LOAD THE DATASET")
    print("=" * 80)
    
    df_raw = "dataset/swiss_reduced_dataset.csv"

    if not os.path.exists(df_raw):
        # build the reduced swiss dataset from PISA 2022 dataset (requires manual download)
        df_raw = reduced_swiss_dataset()
    else:
        df_raw = pd.read_csv(df_raw)

    # =========================================================
    # ========== II. PREPROCESSING PROCESS =====================
    # =========================================================
    
    print("\n" + "=" * 80)
    print("II. PREPROCESSING PROCESS")
    print("=" * 80)
    
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

    # =========================================================
    # ========== III. GLOBAL EXPLORATORY DATA ANALYSIS ========
    # =========================================================
    
    print("\n" + "=" * 80)
    print("III. GLOBAL EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    for dir_path in [OUTPUTS_DIR, TABLES_DIR, PLOTS_DIR, MODELS_DIR, SAVEDFILES_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Data analysis: generate tables
    save_descriptive_statistics(df_raw_with_correct_features, "GLOBAL DATA EXPLORATORY ANALYSIS (BEFORE CLEANING)")
    save_descriptive_statistics(df_preprocessed, "GLOBAL DATA EXPLORATORY ANALYSIS (CLEANED WHOLE DATASET)")
    print("Columns removed from training-set missingness:", removed_missing_columns or "None")
    save_missing_values_report(df_raw_with_correct_features, MISSING_VALUES_THRESHOLD, "MISSING VALUES REPORT (BEFORE CLEANING)")

    # Data analysis: generate plots 
    save_individual_histograms(df_raw_with_correct_features, "BEFORE CLEANING")
    save_individual_violin_plots(df_raw_with_correct_features, "BEFORE CLEANING")
    save_individual_barplots(df_raw_with_correct_features, "BEFORE CLEANING")
    save_combined_histograms(df_raw_with_correct_features)
    save_combined_violin_plots(df_raw_with_correct_features)
    save_combined_barplots(df_raw_with_correct_features)

    # Data analysis: display tables and plots
    print(f"\n DESCRIPTIVE STATISTICS:")
    print_files(f"{TABLES_DIR}/GLOBAL DATA EXPLORATORY ANALYSIS (BEFORE CLEANING).txt")
    print(f"\n COMBINED BAR PLOTS FOR ALL CATEGORICAL FEATURES:")
    display_visualisation(f"{PLOTS_DIR}/univariate_plots/barplots/Bar Plots - All Categorical Features (BEFORE CLEANING AND IMPUTATION).png")
    print(f"\n COMBINED HISTOGRAMS FOR ALL NUMERIC FEATURES:")
    display_visualisation(f"{PLOTS_DIR}/univariate_plots/histograms/Histograms - All Numeric Features (BEFORE CLEANING AND IMPUTATION).png")
    print(f"\n COMBINED VIOLIN PLOTS FOR ALL NUMERIC FEATURES:")
    display_visualisation(f"{PLOTS_DIR}/univariate_plots/violin_plots/Violin Plots - All Numeric Features (BEFORE CLEANING AND IMPUTATION).png")

    # ===========================================================
    # IV. EXPLORATORY DATA ANALYSIS ON TRAIN-SET AFTER IMPUTATION 
    # ===========================================================
    
    print("\n" + "=" * 80)
    print("IV. EXPLORATORY DATA ANALYSIS ON TRAIN-SET AFTER IMPUTATION")
    print("=" * 80)

    eda_preprocessor = create_eda_preprocessor(X_train)
    X_train_imputed_for_eda = eda_preprocessor.fit_transform(X_train)

    df_train_set_for_eda = create_imputed_dataframe(
    X_train_imputed_for_eda,
    eda_preprocessor,
    y_train,
    )

    # Data analysis: generate and save correlation computations
    save_spearman_correlation_matrix(df_train_set_for_eda, "SPEARMAN CORRELATION MATRIX (TRAIN SET AFTER IMPUTATION)")
    save_correlations_with_target(df_train_set_for_eda, "PV1MATH")
    save_correlations_among_features(df_train_set_for_eda, "PV1MATH")

    # Data analysis: display correlation tables and correlation matrix
    display_visualisation(f"{PLOTS_DIR}/multivariate_plots/SPEARMAN CORRELATION MATRIX (TRAIN SET AFTER IMPUTATION).png")
    print_files(f"{TABLES_DIR}/correlations_with_PV1MATH.txt")
    print_files(f"{TABLES_DIR}/correlations_between_features.txt")

    
    # ==============================================================
    # V. FEATURE SELECTION (FOR CLASSIFICATION AND REGRESSION TASKS) 
    # ==============================================================
    
    print("\n" + "=" * 80)
    print("V. FEATURE SELECTION (FOR CLASSIFICATION AND REGRESSION TASKS)")
    print("=" * 80)

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

    #Display feature selection tables
    print_files(f"{TABLES_DIR}/variance_threshold_ranking.txt")
    print("=" * 80)
    print(f"REGRESSION TASK:")
    print("=" * 80)
    print_files(f"{TABLES_DIR}/regression_mutual_info_ranking.txt")
    print_files(f"{TABLES_DIR}/regression_anova_ranking.txt")
    print_files(f"{TABLES_DIR}/regression_rfe_ranking.txt")
    print_files(f"{TABLES_DIR}/regression_combined_feature_selection_rankings.txt")
    print("=" * 80)
    print(f"CLASSIFICATION TASK:")
    print("=" * 80)
    print_files(f"{TABLES_DIR}/classification_mutual_info_ranking.txt")
    print_files(f"{TABLES_DIR}/classification_anova_ranking.txt")
    print_files(f"{TABLES_DIR}/classification_rfe_ranking.txt")
    print_files(f"{TABLES_DIR}/classification_combined_feature_selection_rankings.txt")


    # ==============================================================
    # ========== VI.I MODEL TRAINING FOR REGRESSION TASK  ==========
    # ==============================================================
    
    print("\n" + "=" * 80)
    print("VI.I MODEL TRAINING FOR REGRESSION TASK")
    print("=" * 80)

    # Evaluate regression models for different top-k features
    regression_results_df, regression_all_results_dic = evaluate_k_values(
        X_train_after_var_thresh_reg, y_train, X_test_after_var_thresh_reg, y_test,
        final_features_ranking = final_features_ranking_reg,
        feature_names = selected_columns_var_thresh_reg,
        model_names=['LR', 'RF', 'XGBoost', 'SVR'],
        task='regression',
        k_values=[5, 10, 15, 20]
    )

    # ========== TEMPORARY FILES SAVED FOR QUICKER DEBUG PURPOSE ======================
    # Save regression results DataFrame to tables folder
    #regression_results_df.to_csv(os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv'), index=False)
    #print(f" Regression results saved to '{os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv')}'")

    # Save regression all_results dictionary to models folder
    #joblib.dump(regression_all_results_dic, os.path.join(MODELS_DIR, 'regression_all_results.pkl'))
    #print(f" Regression all_results saved to '{os.path.join(MODELS_DIR, 'regression_all_results.pkl')}'")

    #regression_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv'))
    #regression_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'regression_all_results.pkl'))
    # =================================================================================

    save_detailed_results_by_k(regression_all_results_dic, task='regression')
    save_results(regression_results_df)
    save_summary_of_metrics(regression_results_df, task='regression')

    save_multiple_metrics_vs_features_plot(regression_results_df, task='regression')
    save_residuals_plot_from_all_results(regression_all_results_dic, k_chosen=10)
    save_learning_curves_plot_all_models(
            regression_all_results_dic, 
            k_chosen=10, 
            X_train_after_var_thresh=X_train_after_var_thresh_reg, 
            y_train=y_train, 
            task='regression'
        )

    # ==============================================================
    # ======= VI.II MODEL TRAINING FOR CLASSIFICATION TASK  ========
    # ==============================================================
    
    print("\n" + "=" * 80)
    print("VI.II MODEL TRAINING FOR CLASSIFICATION TASK")
    print("=" * 80)
    
    # Evaluate classification models for different top-k features
    classification_results_df, classification_all_results_dic = evaluate_k_values(
            X_train_after_var_thresh_class, y_train_class, X_test_after_var_thresh_class, y_test_class,
            final_features_ranking = final_features_ranking_class,
            feature_names = selected_columns_var_thresh_class,
            model_names=['LR', 'RF', 'XGBoost', 'SVC'],
            task='classification',
            k_values=[5, 10, 15, 20]
        )

    # Save classification results DataFrame to tables folder
    classification_results_df.to_csv(os.path.join(SAVEDFILES_DIR, 'classification_results_by_k.csv'), index=False)
    print(f"Classification results saved to '{os.path.join(SAVEDFILES_DIR, 'classification_results_by_k.csv')}'")

    # Save classification all_results dictionary to models folder
    joblib.dump(classification_all_results_dic, os.path.join(MODELS_DIR, 'classification_all_results.pkl'))
    print(f"Classification all_results saved to '{os.path.join(MODELS_DIR, 'classification_all_results.pkl')}'")
    
    classification_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'classification_results_by_k.csv'))
    classification_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'classification_all_results.pkl'))

    save_detailed_results_by_k(classification_all_results_dic, task='classification')
    save_results(classification_results_df)
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
    

    # 6. ========= EXPLAINABILITY ===========
    
    # RASHOMON SET BUILDER
    classification_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'classification_results_by_k.csv'))
    classification_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'classification_all_results.pkl'))
    regression_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv'))
    regression_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'regression_all_results.pkl'))
    
    # Build the Rashomon set based on the k-features chosen to train models
    k_chosen = 10
    rashomon_set, rashomon_best_score, rashomon_lowest_score_acceptable = rashomon_set_builder(regression_all_results_dic, k_nbr_of_features_chosen= k_chosen, task='regression', rashomon_threshold=0.05)
    save_rashomon_set(rashomon_set, rashomon_best_score, rashomon_lowest_score_acceptable, task='regression', k_nbr_of_features_chosen= k_chosen, rashomon_threshold=0.05)
    
    # Create X_train/X_test for the related k-features selected
    indices_k = regression_all_results_dic[k_chosen]['indices']
    feature_names_k = regression_all_results_dic[k_chosen]['features']
    X_train_k = X_train_after_var_thresh_reg[:, indices_k]
    X_test_k = X_test_after_var_thresh_reg[:, indices_k]

    # Compute global shap values
    shap_cache_path = os.path.join(SAVEDFILES_DIR, 'shap_results_cache.joblib')

    #To disable when launched
    global_shap_rankings, shap_values_for_all_models = local_and_global_shap_values_calculator(rashomon_set, X_train_k, X_test_k, feature_names_k)
    joblib.dump({
            "rankings": global_shap_rankings,
            "shap_values": shap_values_for_all_models
        }, shap_cache_path)
    
    cached_shap_data = joblib.load(shap_cache_path)
    global_shap_rankings = cached_shap_data["rankings"]
    shap_values_for_all_models = cached_shap_data["shap_values"]

    save_global_shap_rankings(global_shap_rankings, k_chosen)
    #print_files("outputs/tables/global_shap_rankings.txt")

    barplot_global_shap_rankings(global_shap_rankings, k_chosen)
    shap_summary_plot(shap_values_for_all_models, X_test_k, feature_names_k)

    inter_model_concordance_df = inter_model_concordance_assessment(global_shap_rankings, top_k_features_concordance=5)
    save_inter_model_concordance_agreement(inter_model_concordance_df)

    feature_agreement_stats_df = feature_agreement_stats(global_shap_rankings)
    save_feature_agreement_stats(feature_agreement_stats_df)

    intra_model_assessment_result = intra_model_stability_assessment(shap_values_for_all_models, feature_names_k)
    save_intra_model_stability_assessment(intra_model_assessment_result)

    final_classification = assess_features_robustness(global_shap_rankings, intra_model_assessment_result, top_k=5)
    save_feature_robustness_assessment(final_classification)
    
    """
    # ======= SHAP ANALYSIS - COMPLETE AND CLEAN PIPELINE ========
    global_shap_rankings, shap_values_for_all_models, X_test_k, feature_names_k, number_of_features_chosen_for_model_training = run_complete_shap_analysis_and_classification(
        X_train = X_train_after_var_thresh_reg,
        X_test = X_test_after_var_thresh_reg,
        number_of_features_chosen_for_model_training = 10,
        task ='regression',
        all_training_model_results_dic = regression_all_results_dic,
        rashomon_set_threshold = 0.05,
        number_of_features_retained_for_final_classification = 5,
        )

    print_files("outputs/tables/rashomon_set.txt")
    print_files("outputs/tables/global_shap_rankings.txt")
    barplot_global_shap_rankings(global_shap_rankings, number_of_features_chosen_for_model_training)
    shap_summary_plot(shap_values_for_all_models, X_test_k, feature_names_k)
    print_files("outputs/tables/feature_agreement_stats.txt")
    print_files("outputs/tables/inter_model_concordance_agreement.txt")
    print_files("outputs/tables/intra_model_stability_assessment.txt")
    print_files("outputs/tables/feature_robustness_assessment.txt")

    # ===============================================================================================
    """
    