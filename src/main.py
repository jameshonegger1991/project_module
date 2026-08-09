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
    display_correlations_between_features,
    display_correlations_with_target,
    display_detailed_results_by_k,
    display_results,
    display_summary,
    display_global_shap_rankings,
    generate_missing_values_report,
    get_descriptive_statistics,
)
from src.visualisations import (
    barplot_global_shap_rankings,
    display_barplots,
    display_histograms,
    display_spearman_correlation_matrix,
    display_violin_plots,
    export_barplots,
    export_histograms,
    export_violin_plots,
    plot_confusion_matrices_all_models,
    plot_learning_curves_all_models,
    plot_multiple_metrics_vs_features,
    plot_residuals_from_all_results,
    shap_summary_plot,
)
from src.model_training import evaluate_k_values, run_models
from src.SHAP_analysis import (
    local_and_global_shap_values_calculator, 
    rashomon_set_builder, 
    inter_model_concordance_assessment, 
    feature_agreement_stats,
    intra_model_stability_assessment,
)








if __name__ == "__main__":

    # 1. ========== LOAD / BUILD DATASET ==========
    df_raw = reduced_swiss_dataset()

    (X_train_imputed, X_test_imputed, X_train, X_test, y_train, y_test, df_raw_with_correct_features, df_preprocessed, numeric_imputed_feature_names, all_imputed_feature_names, removed_missing_columns) = run_preprocessing_pipeline(df_raw, MISSING_VALUES_THRESHOLD)

    for dir_path in [OUTPUTS_DIR, TABLES_DIR, PLOTS_DIR, MODELS_DIR, SAVEDFILES_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    """
    # 2. ========== GLOBAL EXPLORATORY DATA ANALYSIS ==========
    #get_descriptive_statistics(df_raw_with_correct_features, "GLOBAL DATA EXPLORATORY ANALYSIS (BEFORE CLEANING)")
    #get_descriptive_statistics(df_preprocessed, "GLOBAL DATA EXPLORATORY ANALYSIS (CLEANED WHOLE DATASET)")
    #print()
    #print("Columns removed from training-set missingness:", removed_missing_columns or "None")
    #print()
    #generate_missing_values_report(df_raw_with_correct_features, MISSING_VALUES_THRESHOLD, "MISSING VALUES REPORT (BEFORE CLEANING)")

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
    """
    #y_train/test for classification task
    y_train_class = pd.cut(y_train, bins = CLASS_BOUNDARIES, labels = CLASS_LABELS, right = False, include_lowest = True)
    y_test_class = pd.cut(y_test,bins = CLASS_BOUNDARIES, labels = CLASS_LABELS, right = False, include_lowest = True)

    # regression
    (final_features_ranking_reg, variance_threshold_df_reg, ranking_MI_df_reg, ranking_anova_df_reg, ranking_rfe_df_reg, X_train_after_var_thresh_reg, X_test_after_var_thresh_reg, selected_columns_var_thresh_reg) = run_feature_selection_pipeline(X_train_imputed, X_test_imputed, y_train, all_imputed_feature_names, task = "regression")
    """
    #classification
    (final_features_ranking_class, variance_threshold_df_class, ranking_MI_df_class, ranking_anova_df_class, ranking_rfe_df_class, X_train_after_var_thresh_class, X_test_after_var_thresh_class, selected_columns_var_thresh_class) = run_feature_selection_pipeline(X_train_imputed, X_test_imputed, y_train_class, all_imputed_feature_names, task = "classification")
    
    # 5. ========= MODEL TRAINING ===========
    
    print("\n" + "=" * 80)
    print("REGRESSION MODELS EVALUATION")
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
    
    # Save regression results DataFrame to tables folder
    regression_results_df.to_csv(os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv'), index=False)
    print(f" Regression results saved to '{os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv')}'")

    # Save regression all_results dictionary to models folder
    joblib.dump(regression_all_results_dic, os.path.join(MODELS_DIR, 'regression_all_results.pkl'))
    print(f" Regression all_results saved to '{os.path.join(MODELS_DIR, 'regression_all_results.pkl')}'")

    regression_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv'))
    regression_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'regression_all_results.pkl'))

    display_detailed_results_by_k(regression_all_results_dic)
    display_summary(regression_results_df, task='regression')

    plot_multiple_metrics_vs_features(regression_results_df, task='regression')
    plot_residuals_from_all_results(regression_all_results_dic, k_chosen=10)
    plot_learning_curves_all_models(
            regression_all_results_dic, 
            k_chosen=10, 
            X_train_after_var_thresh=X_train_after_var_thresh_reg, 
            y_train=y_train, 
            task='regression'
        )
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODELS EVALUATION")
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

    display_detailed_results_by_k(classification_all_results_dic)
    display_summary(classification_results_df, task='classification')

    plot_multiple_metrics_vs_features(classification_results_df, task='classification')
    plot_confusion_matrices_all_models(classification_all_results_dic, k_chosen=10, class_labels=CLASS_LABELS)

    plot_learning_curves_all_models(
        classification_all_results_dic, 
        k_chosen=10, 
        X_train_after_var_thresh=X_train_after_var_thresh_class, 
        y_train=y_train_class, 
        task='classification'
    )
    """

    # 6. ========= EXPLAINABILITY ===========

    # RASHOMON SET BUILDER
    classification_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'classification_results_by_k.csv'))
    classification_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'classification_all_results.pkl'))
    regression_results_df = pd.read_csv(os.path.join(SAVEDFILES_DIR, 'regression_results_by_k.csv'))
    regression_all_results_dic = joblib.load(os.path.join(MODELS_DIR, 'regression_all_results.pkl'))

    # Build the Rashomon set based on the k-features chosen to train models
    k_chosen = 10
    rashomon_set = rashomon_set_builder(regression_all_results_dic, k_nbr_of_features_chosen= k_chosen, task='regression', rashomon_threshold=0.05)

    # Create X_train/X_test for the related k-features selected
    indices_k = regression_all_results_dic[k_chosen]['indices']
    feature_names_k = regression_all_results_dic[k_chosen]['features']
    X_train_k = X_train_after_var_thresh_reg[:, indices_k]
    X_test_k = X_test_after_var_thresh_reg[:, indices_k]

    # Compute global shap values
    shap_cache_path = os.path.join(SAVEDFILES_DIR, 'shap_results_cache.joblib')

    """
    global_shap_rankings, shap_values_for_all_models = local_and_global_shap_values_calculator(rashomon_set, X_train_k, X_test_k, feature_names_k)
    joblib.dump({
            "rankings": global_shap_rankings,
            "shap_values": shap_values_for_all_models
        }, shap_cache_path)
    """
    cached_shap_data = joblib.load(shap_cache_path)
    global_shap_rankings = cached_shap_data["rankings"]
    shap_values_for_all_models = cached_shap_data["shap_values"]

    display_global_shap_rankings(global_shap_rankings, k_chosen)
    barplot_global_shap_rankings(global_shap_rankings, k_chosen)
    shap_summary_plot(shap_values_for_all_models, X_test_k, feature_names_k)
    inter_model_concordance_df = inter_model_concordance_assessment(global_shap_rankings, top_k_features_concordance=5)
    print(inter_model_concordance_df)
    feature_agreement_stats_df = feature_agreement_stats(global_shap_rankings)
    print(feature_agreement_stats_df)
    print()
    intra_model_assessment_result = intra_model_stability_assessment(shap_values_for_all_models, feature_names_k)
    print(intra_model_assessment_result)