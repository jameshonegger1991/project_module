import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_column
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_row
from src.config import VARIANCE_THRESHOLD
from src.utils import check_overfitting_regression, check_overfitting_classification


def save_missing_values_report(df, threshold, title):
    """
    Display missing-value statistics for columns and rows.
    """
    
    col_pct = get_missing_percentages_per_column(df)
    row_pct = get_missing_percentages_per_row(df)

    cols_above_threshold = col_pct[col_pct > threshold]
    cols_above_threshold_df = pd.DataFrame({'Column': cols_above_threshold.index,'Missing %': cols_above_threshold.values}).sort_values(by = 'Missing %', ascending = False)

    total_rows = len(df)
    rows_missing_50 = (row_pct >= 50).sum()
    rows_missing_70 = (row_pct >= 70).sum()
    rows_missing_90 = (row_pct >= 90).sum()
    rows_above_threshold = row_pct[row_pct > threshold]

    rows_with_missing = row_pct[row_pct > 0]  
    mean_missing_pct = rows_with_missing.mean() if len(rows_with_missing) > 0 else 0
    median_missing_pct = rows_with_missing.median() if len(rows_with_missing) > 0 else 0
    rows_with_missing_count = len(rows_with_missing)

    file_path = f"outputs/tables/{title}.txt"
    save_file = open(file_path, "w")
    print(file = save_file)
    print("=" * 80, file = save_file)
    print(f"{title}", file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    print("\n=== COLUMNS ===", file = save_file)
    print(f"Total columns: {len(col_pct)}", file = save_file)
    print(f"Columns with > {threshold}% missing: {len(cols_above_threshold)}", file = save_file)
    if len(cols_above_threshold) > 0:
        print(f"\nColumns with missingness above {threshold}%:", file = save_file)
        print(cols_above_threshold_df.to_string(index=False), file = save_file)
    
    print("\n=== ROWS ===", file = save_file)
    print(f"Total rows: {total_rows:,}", file = save_file)
    print(f"Rows with > {threshold}% missing: {len(rows_above_threshold):,} ({len(rows_above_threshold)/total_rows*100:.1f}%)", file = save_file)
    print(f"Rows with ≥ 50% missing: {rows_missing_50:,} ({rows_missing_50/total_rows*100:.1f}%)", file = save_file)
    print(f"Rows with ≥ 70% missing: {rows_missing_70:,} ({rows_missing_70/total_rows*100:.1f}%)", file = save_file)
    print(f"Rows with ≥ 90% missing: {rows_missing_90:,} ({rows_missing_90/total_rows*100:.1f}%)", file = save_file)
    
    print(f"\n=== Rows with missing values ===", file = save_file)
    print(f"Rows with at least one missing value: {rows_with_missing_count:,} ({rows_with_missing_count/total_rows*100:.1f}%)", file = save_file)
    if rows_with_missing_count > 0:
        print(f"Mean missing % per row (among rows with missing): {mean_missing_pct:.1f}%", file = save_file)
        print(f"Median missing % per row (among rows with missing): {median_missing_pct:.1f}%", file = save_file)
        print(f"Average missing cells per row: {(rows_with_missing / 100 * len(df.columns)).mean():.1f} cells", file = save_file)


    print("\n=== SUMMARY ===", file = save_file)
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    print(f"Total cells: {total_cells:,}", file = save_file)
    print(f"Total missing cells: {total_missing:,}", file = save_file)
    print(f"Overall missing %: {(total_missing / total_cells) * 100:.1f}%", file = save_file)
    print("=" * 80 + "\n", file = save_file)
    save_file.close()
    print(f"{title}.txt saved to {file_path}")

def save_descriptive_statistics(df, title):
    """
    Display descriptive statistics for a DataFrame.
    It includes shape, descriptive stats, head, info, and skewness.
    """
    file_path = f"outputs/tables/{title}.txt"
    save_file = open(file_path, "w")
    
    print("=" * 80, file = save_file)
    print(title, file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    
    print("=" * 80, file = save_file)
    print("SHAPE OF THE DATASET", file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    print(df.shape, file = save_file)
    print(file = save_file)

    print("=" * 80, file = save_file)
    print("HEAD OF THE DATASET", file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    print(df.head(), file = save_file)
    print(file = save_file)
    
    print("=" * 80, file = save_file)
    print("DESCRIPTIVE STATISTICS", file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print(numeric_df.describe(), file = save_file)
    print(file = save_file)
    
    print("=" * 80, file = save_file)
    print("DATA TYPES", file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    type_summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    print(type_summary.to_string(index=False), file = save_file)
    print(file = save_file)
    
    print("=" * 80, file = save_file)
    print("SKEWNESS OF THE DATASET",file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    print(numeric_df.skew(), file = save_file)
    print(file = save_file)
    
    print("=" * 80 + "\n", file = save_file)
    save_file.close()
    print(f"{title}.txt saved to {file_path}")

def save_correlations_with_target(df, target = "PV1MATH"):
    """
    Displays all variables according to their Spearman correlation
    with the target variable.

    Positive and negative correlations are displayed separately.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target (str): The target variable.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot calculate correlations.")
        return

    if target not in df.columns:
        print(f"The target variable '{target}' was not found.")
        return

    correlations = (df.corr(method="spearman")[target].drop(target))
    positive_correlations = (correlations[correlations > 0].sort_values(ascending=False).rename("Correlation").to_frame())
    positive_correlations.index.name = "Feature"

    negative_correlations = (correlations[correlations < 0].sort_values(ascending=True).rename("Correlation").to_frame())

    negative_correlations.index.name = "Feature"

    file_path = f"outputs/tables/correlations_with_{target}.txt"
    save_file = open(file_path, "w")
    print("\n" + "=" * 80,file = save_file)
    print("CORRELATIONS WITH TARGET",file = save_file)
    print("=" * 80,file = save_file)
    print(f"\nPOSITIVE CORRELATIONS WITH {target}", file = save_file)
    print(positive_correlations, file = save_file)

    print(f"\nNEGATIVE CORRELATIONS WITH {target}", file = save_file)
    print(negative_correlations, file = save_file)
    save_file.close()
    print(f"correlations_with_{target}.txt saved to {file_path}")

def save_correlations_among_features(df, target = "PV1MATH", corr_threshold = 0.70):
    """
    Displays the 20 strongest positive and negative Spearman correlations
    among features, excluding the target variable.

    It also displays feature pairs whose absolute correlation exceeds
    the specified threshold, indicating a potential risk of multicollinearity.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot calculate correlations.")
        return

    features = df.drop(columns=[target], errors="ignore")
    correlation_matrix = features.corr(method="spearman")
    correlations = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            correlations.append({
                "Feature 1": correlation_matrix.columns[i],
                "Feature 2": correlation_matrix.columns[j],
                "Correlation": correlation_matrix.iloc[i, j]
            })

    correlations_df = pd.DataFrame(correlations).dropna()

    # Selection of  the 20 strongest positive correlations.
    positive_correlations = (
        correlations_df[correlations_df["Correlation"] > 0]
        .sort_values("Correlation", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # Selection of the 20 strongest negative correlations.
    negative_correlations = (
        correlations_df[correlations_df["Correlation"] < 0]
        .sort_values("Correlation", ascending=True)
        .head(20)
        .reset_index(drop=True)
    )

    # Select correlations exceeding the multicollinearity threshold.
    # The literature recommends a threshold between 0.7 - 0.8 
    # SOURCE:
    # - Arroyo Resino et al. - 2024 - Student well-being and mathematical literacy performance in PISA 2018 a machine-learning approach.
    # - Masci et al. - 2018 - Student and school performance across countries A machine learning approach.
    multicollinearity_risk = (
        correlations_df[
            correlations_df["Correlation"].abs() >= corr_threshold
        ]
        .assign( # Using .assign() avoids a SettingWithCopyWarning by returning a new DataFrame.
            Absolute_Correlation = lambda x: x["Correlation"].abs() # INSPIRATION: https://medium.com/@whyamit404/understanding-pandas-assign-with-step-by-step-examples-6cf2d79ff481 
        )
        .sort_values("Absolute_Correlation", ascending=False)
        .reset_index(drop=True)
    )

    file_path = f"outputs/tables/correlations_among_features.txt"
    save_file = open(file_path, "w")
    print("\n" + "=" * 80, file = save_file)
    print("CORRELATIONS BETWEEN FEATURES", file = save_file)
    print("=" * 80, file = save_file)
    print("\n20 STRONGEST POSITIVE CORRELATIONS BETWEEN FEATURES", file = save_file)
    print(positive_correlations, file = save_file)

    print("\n20 STRONGEST NEGATIVE CORRELATIONS BETWEEN FEATURES", file = save_file)
    print(negative_correlations, file = save_file)

    print(f"\nPOTENTIAL MULTICOLLINEARITY RISK ", file = save_file)
    print(f"(|CORRELATION| >= {corr_threshold})", file = save_file)

    if multicollinearity_risk.empty:
        print("No feature pair exceeds the selected corr_threshold.", file = save_file)
    else:
        print(multicollinearity_risk, file = save_file)
    save_file.close()
    print(f"correlations_among_features.txt saved to {file_path}")

# ==== FEATURE SELECTION ==== 

def save_variance_threshold(ranking_VThresh: pd.DataFrame, threshold = VARIANCE_THRESHOLD, selected_cols: list = [], feature_names: list = []):
    save_file = open(f"outputs/tables/variance_threshold_ranking.txt", "w")
    print("\n" + "=" * 80, file = save_file)
    print(f"Variance Threshold (threshold = {threshold}) : {len(selected_cols)}/{len(feature_names)} kept", file = save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)
    print(ranking_VThresh.to_string(index=False), file = save_file)
    print(file = save_file)
    save_file.close()
    print("variance_threshold_ranking.txt saved to outputs/tables")

def save_mutual_info(ranking_MI: pd.DataFrame, task, X_train):

    save_file = open(f"outputs/tables/{task}_mutual_info_ranking.txt", "w")
    print("\n" + "=" * 80, file = save_file)
    print(f"Mutual Information ({task}) : Complete ranking of {X_train.shape[1]} features.", file = save_file)
    print("=" * 80, file = save_file)
    print("\n Mutual Information Ranking (1 = best) :", file = save_file)
    print(file = save_file)
    print(ranking_MI.to_string(index=False), file = save_file)
    print(file = save_file)
    save_file.close()
    print(f"{task}_mutual_info_ranking.txt saved to outputs/tables")
    
def save_anova(ranking_anova: pd.DataFrame, task, X_train):

    save_file = open(f"outputs/tables/{task}_anova_ranking.txt", "w")
    print("\n" + "=" * 80, file = save_file)
    print(f"ANOVA ({task}) : Complete ranking of {X_train.shape[1]} features.", file = save_file)
    print("=" * 80, file = save_file)
    print("\n ANOVA Ranking (1 = best F-score) :", file = save_file)
    print(ranking_anova.to_string(index = False), file = save_file)
    save_file.close()
    print(f"{task}_anova_ranking.txt saved to outputs/tables")

def save_rfe(ranking_rfe: pd.DataFrame, task, X_train):

    save_file = open(f"outputs/tables/{task}_rfe_ranking.txt", "w")
    print("\n" + "=" * 80, file = save_file)
    print(f"RFE ({task}) : Complete ranking of {X_train.shape[1]} features.", file = save_file)
    print("=" * 80, file = save_file)
    print("\n RFE Ranking (1 = best) :", file = save_file)
    print(ranking_rfe.to_string(index=False), file = save_file)
    save_file.close()
    print(f"{task}_rfe_ranking.txt saved to outputs/tables")

def save_summarised_feature_rankings(combined_rankings: pd.DataFrame, method_names = None, task = 'regression'):

    save_file = open(f"outputs/tables/{task}_combined_feature_selection_rankings.txt", "w")
    print("\n" + "=" * 80, file = save_file)
    print("COMBINED FINAL RANKING", file = save_file)
    print("=" * 80, file = save_file)
    if method_names is not None:
        print(f"\n {len(method_names)} combined methods: {', '.join(method_names)}", file = save_file)
    print("\n Final Ranking:", file = save_file)
    print(combined_rankings.to_string(index=False), file = save_file)
    save_file.close()
    print(f"{task}_combined_feature_selection_rankings.txt saved to outputs/tables")
    
def save_top_features(top_features: list, k, save_file=None, task = 'regression'):

    own_file = save_file is None

    if own_file:
        save_file = open(f"outputs/tables/{task}_top_{k}_features_selected.txt", "w")

    print("\n" + "=" * 80, file = save_file)
    print(f"TOP {k} FEATURES SELECTED FOR {task.upper()} TASK", file=save_file)
    print("=" * 80, file = save_file)
    print(file = save_file)

    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature}", file=save_file)

    if own_file:
        save_file.close()


# ==== MODEL TRAINING ====

def save_results(results, model_names=None, top_k=10, save_file = None):
    """
    Save results for all models in the results dictionary.
    """
    if model_names is None:
        model_names = list(results.keys())

    for name in model_names:
        if name not in results:
            print(f"Model '{name}' not found in results.")
            continue

        result = results[name]

        if result['type'] == 'regression':
            save_regression_result(name, result, top_k, save_file)
        else:
            save_classification_result(name, result, top_k, save_file)

def save_regression_result(name, result, top_k, save_file = None):
    """Internal function to save a single regression result."""

    own_file = save_file is None
    if own_file:
        save_file = open(f"outputs/tables/{name}_regression_results_{top_k}_features.txt","w")

    print("=" * 50, file = save_file)
    print(f"{name} (REGRESSION) - RESULTS", file = save_file)
    print("=" * 50, file = save_file)
    print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}", file = save_file)
    print("-" * 50, file = save_file)
    print(f"{'R²':<12} {result['train_r2']:>10.4f} {result['test_r2']:>10.4f} {result['gap_r2']:>10.4f}", file = save_file)
    print(f"{'MSE':<12} {result['train_mse']:>10.2f} {result['test_mse']:>10.2f} {result['train_mse'] - result['test_mse']:>10.2f}", file = save_file)
    print(f"{'RMSE':<12} {result['train_rmse']:>10.4f} {result['test_rmse']:>10.4f} {result['train_rmse'] - result['test_rmse']:>10.4f}", file = save_file)
    print(f"{'MAE':<12} {result['train_mae']:>10.4f} {result['test_mae']:>10.4f} {result['train_mae'] - result['test_mae']:>10.4f}", file = save_file)
    print("-" * 50, file = save_file)
    
    if result.get('is_lasso'):
        print(f"Best alpha        : {result['best_alpha']}", file = save_file)
        print(f"Features selected : {result['n_selected']}/{result['p']}", file = save_file)

    if 'best_cv_score' in result:
        print(f"Best CV R²-score  : {result['best_cv_score']:.4f}", file = save_file)

    # Overfitting status
    cv_score = result.get('best_cv_score', result['test_r2'])
    status, detail, _ = check_overfitting_regression(result['train_r2'], cv_score)
    print(f"\n{status}: {detail}", file = save_file)
    print(file = save_file)

    if own_file:
        save_file.close()
        print(f"{name}_regression_results_{top_k}_features.txt saved to outputs/tables")

def save_classification_result(name, result, top_k, save_file = None):
    """Internal function to save a single classification result."""

    own_file = save_file is None

    if own_file:
        save_file = open(f"outputs/tables/{name}_classifications_result_{top_k}_features.txt","w")

    print("=" * 50, file = save_file)
    print(f"{name} (CLASSIFICATION) - RESULTS", file = save_file)
    print("=" * 50, file = save_file)
    print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}", file = save_file)
    print("-" * 50, file = save_file)
    print(f"{'Accuracy':<15} {result['train_accuracy']:>10.4f} {result['test_accuracy']:>10.4f} {result['gap_accuracy']:>10.4f}", file = save_file)
    print(f"{'F1 (macro)':<15} {result['train_f1_macro']:>10.4f} {result['test_f1_macro']:>10.4f} {result['gap_f1']:>10.4f}", file = save_file)
    print("-" * 50, file = save_file)
    
    # Classification report with original labels
    print(f"\nClassification Report {name} (Test):", file = save_file)
    print(classification_report(result['y_test_true'], result['y_pred']), file = save_file)
    print("\nConfusion Matrix (Test):", file = save_file)
    print(confusion_matrix(result['y_test_true'], result['y_pred']), file = save_file)
    print(file = save_file)
    print(f"Best parameters: {result['best_params']}", file = save_file)
    print(f"Best CV F1-score: {result['best_cv_score']:.4f}", file = save_file)
    
    # Overfitting status
    status, detail, _ = check_overfitting_classification(result['train_f1_macro'], result['best_cv_score'])
    print(f"\n{status}: {detail}", file = save_file)
    print(file = save_file)

    if own_file:
        save_file.close()
        print(f"{name}_classification_results_{top_k}_features.txt saved to outputs/tables")

def save_detailed_results_by_k(all_results, task='regression', k_values=None):
    """
    Save detailed results for each k in a single text file.
    """

    save_file = open(f"outputs/tables/{task}_detailed_results_by_k.txt","w")

    if k_values is None:
        k_values = sorted(all_results.keys())

    for k in k_values:

        print(f"\n{'#' * 60}", file=save_file)
        print(f"# DETAILED RESULTS FOR K = {k} FEATURES", file=save_file)
        print(f"{'#' * 60}", file=save_file)

        features = all_results[k]['features']
        save_top_features(features, k, save_file=save_file, task=task)

        results_k = all_results[k]['results']
        save_results(results_k, top_k=k, save_file=save_file)

    save_file.close()
    print(f"{task}_detailed_results_by_k.txt saved to outputs/tables")

def save_summary_of_metrics(results_df, task):
    """
    Saves a summary of metrics from results_df.
    """
    save_file = open(f"outputs/tables/{task}_summary_of_metrics.txt", "w")
    print("\n" + "=" * 80, file = save_file)
    print(f"SUMMARY - {task.upper()} METRICS", file = save_file)
    print("=" * 80, file = save_file)
    
    if task == 'regression':
        display_cols = ['k', 'Model', 'CV_Score', 'R²_test', 'RMSE_test', 'MAE_test', 'Gap_R²']
        if 'Features_Selected' in results_df.columns:
            display_cols.append('Features_Selected')
        print(results_df[display_cols].to_string(index=False), file = save_file)
    else:
        print(results_df[['k', 'Model', 'CV_Score', 'Accuracy_test', 'F1_macro_test', 'Gap_Accuracy']].to_string(index=False), file = save_file)

    save_file.close()

    print(f"{task}_summary_of_metrics.txt saved to outputs/tables")

    

###### SHAP ANALYSIS #####

def save_global_shap_rankings(global_mean_absolute_shap_rankings_dict, top_n=10):

    save_file = open('outputs/tables/global_shap_rankings.txt', 'w')

    for model_name, ranking_df in global_mean_absolute_shap_rankings_dict.items():

        
        print(f"\n{'='*60}", file = save_file)
        print(f"SHAP FEATURE IMPORTANCE RANKING : {model_name}, top {top_n} features", file = save_file)
        print(f"{'='*60}", file = save_file)
        print(ranking_df.head(top_n).to_string(index=False), file = save_file)
        print("="*60 + "\n", file = save_file)

    save_file.close()

def save_rashomon_set(rashomon_set_dict, rashomon_best_score, rashomon_lowest_score_acceptable, task='regression', k_nbr_of_features_chosen=None, rashomon_threshold=None):
    save_file = open('outputs/tables/rashomon_set.txt', 'w')

    metric_label = "best R² CV score" if task == 'regression' else "best F1 macro CV score"
    
    print(f"\n{'='*80}", file = save_file)
    print(f"RASHOMON SET CONSTRUCTION (k={k_nbr_of_features_chosen}, Task={task})", file = save_file)
    print(f"{'='*80}", file = save_file)
    print(f"\nBest CV Score achieved : {rashomon_best_score:.4f}", file = save_file)
    print(f"Acceptable threshold   : >= {rashomon_lowest_score_acceptable:.4f} (Δ <= {rashomon_threshold})", file = save_file)
    print(f"\nRashomon set contains the following models:", file = save_file)
    print(file = save_file)
    for model_name, res_dict in rashomon_set_dict.items():
        score = res_dict.get('best_cv_score')
        print(f"  • {model_name:<10} : {metric_label} = {score:.4f}", file = save_file)
        
    print(f"\n{'='*80}\n", file = save_file)
    save_file.close()

def save_inter_model_concordance_agreement(concordance_dict):

    save_file = open('outputs/tables/inter_model_concordance_agreement.txt', 'w')
    print(f"\n{'='*100}", file = save_file)
    print(f"INTER-MODEL CONCORDANCE AGREEMENT", file = save_file)
    print(f"{'='*100}", file = save_file)
    print(file = save_file)
    print(concordance_dict, file = save_file)
    print(f"\n{'='*100}\n", file = save_file)
    save_file.close()

def save_feature_agreement_stats(feature_agreement_stats_df):
    save_file = open('outputs/tables/feature_agreement_stats.txt', 'w')
    
    print(f"{'='*120}", file=save_file)
    print(f"FEATURE AGREEMENT STATS", file=save_file)
    print(f"{'='*120}", file=save_file)
    print(file=save_file)
    
    col_width = 12
    spacing = "     "
    
    feature_col_width = col_width + len(spacing)  
    header_line1 = f"{'Feature':<{feature_col_width}}"  

    for col in feature_agreement_stats_df.columns:

        if ' SHAP RANKING' in col:
            model_name = col.replace(' SHAP RANKING', '')
            header_line1 += f"{model_name:>{col_width}}{spacing}"

        elif col == 'Mean SHAP rank':
            header_line1 += f"{'Mean':>{col_width}}{spacing}"

        elif col == 'Standard deviation SHAP rank':
            header_line1 += f"{'Std dev':>{col_width}}{spacing}"

    print(header_line1, file=save_file)
    

    header_line2 = f"{'':<{feature_col_width}}"
    for col in feature_agreement_stats_df.columns:

        if ' SHAP RANKING' in col:
            header_line2 += f"{'SHAP RANKING':>{col_width}}{spacing}"

        elif col == 'Mean SHAP rank':
            header_line2 += f"{'SHAP rank':>{col_width}}{spacing}"

        elif col == 'Standard deviation SHAP rank':
            header_line2 += f"{'SHAP rank':>{col_width}}{spacing}"

    print(header_line2, file=save_file)
    
    for _, row in feature_agreement_stats_df.iterrows():
        line = f"{row['Feature']:>{col_width + 5}}"  
        for col in feature_agreement_stats_df.columns:

            if col != 'Feature':

                if isinstance(row[col], float):
                    line += f"{row[col]:>{col_width}.6f}{spacing}"

                else:
                    line += f"{row[col]:>{col_width}}{spacing}"

        print(line, file=save_file)
    
    print(f"\n{'='*120}\n", file=save_file)
    save_file.close()

def save_intra_model_stability_assessment(intra_model_assessment_result):

    save_file = open('outputs/tables/intra_model_stability_assessment.txt', 'w')
    
    print(f"{'='*100}", file=save_file)
    print(f"INTRA-MODEL STABILITY ASSESSMENT (Bootstrap Results)", file=save_file)
    print(f"{'='*100}", file=save_file)
    print(file=save_file)
    
    for model_name, df in intra_model_assessment_result.items():
        print(f"\n{model_name}:", file=save_file)
        print("-" * 100, file=save_file)
        print(df.to_string(index=False), file=save_file)
        print(file=save_file)
    
    print(f"\n{'='*100}\n", file=save_file)
    save_file.close()

def save_feature_robustness_assessment(feature_robustness_result):

    save_file = open('outputs/tables/feature_robustness_assessment.txt', 'w')   
    print(f"{'='*100}", file=save_file)
    print(f"FEATURE ROBUSTNESS ASSESSMENT", file=save_file)
    print(f"{'='*100}", file=save_file)
    print(file=save_file)
    print(feature_robustness_result, file=save_file)
    print(f"\n{'='*100}\n", file=save_file)
    save_file.close()