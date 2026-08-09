import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_column
from src.dataset_cleaning_and_preprocessing import get_missing_percentages_per_row
from src.config import VARIANCE_THRESHOLD


def generate_missing_values_report(df, threshold, title):
    """
    Generate a comprehensive missing values report for both columns and rows.
    
    Parameters:
        df: DataFrame to analyse
        threshold: Percentage threshold for highlighting high missingness (default: 70%)
    """
    col_pct = get_missing_percentages_per_column(df)
    row_pct = get_missing_percentages_per_row(df)

    #1. Columns with missing rate > threshold
    cols_above_threshold = col_pct[col_pct > threshold]
    cols_above_threshold_df = pd.DataFrame({'Column': cols_above_threshold.index,'Missing %': cols_above_threshold.values}).sort_values(by = 'Missing %', ascending = False)

    #2. Rows statistics
    total_rows = len(df)
    rows_missing_50 = (row_pct >= 50).sum()
    rows_missing_70 = (row_pct >= 70).sum()
    rows_missing_90 = (row_pct >= 90).sum()
    rows_above_threshold = row_pct[row_pct > threshold]

    rows_with_missing = row_pct[row_pct > 0]  
    mean_missing_pct = rows_with_missing.mean() if len(rows_with_missing) > 0 else 0
    median_missing_pct = rows_with_missing.median() if len(rows_with_missing) > 0 else 0
    rows_with_missing_count = len(rows_with_missing)

    #3. Display
    
    print()
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    print()
    print("\n=== COLUMNS ===")
    print(f"Total columns: {len(col_pct)}")
    print(f"Columns with > {threshold}% missing: {len(cols_above_threshold)}")
    if len(cols_above_threshold) > 0:
        print(f"\nColumns with missingness above {threshold}%:")
        print(cols_above_threshold_df.to_string(index=False))
    
    print("\n=== ROWS ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with > {threshold}% missing: {len(rows_above_threshold):,} ({len(rows_above_threshold)/total_rows*100:.1f}%)")
    print(f"Rows with ≥ 50% missing: {rows_missing_50:,} ({rows_missing_50/total_rows*100:.1f}%)")
    print(f"Rows with ≥ 70% missing: {rows_missing_70:,} ({rows_missing_70/total_rows*100:.1f}%)")
    print(f"Rows with ≥ 90% missing: {rows_missing_90:,} ({rows_missing_90/total_rows*100:.1f}%)")
    
    print(f"\n=== Rows with missing values ===")
    print(f"Rows with at least one missing value: {rows_with_missing_count:,} ({rows_with_missing_count/total_rows*100:.1f}%)")
    if rows_with_missing_count > 0:
        print(f"Mean missing % per row (among rows with missing): {mean_missing_pct:.1f}%")
        print(f"Median missing % per row (among rows with missing): {median_missing_pct:.1f}%")
        print(f"Average missing cells per row: {(rows_with_missing / 100 * len(df.columns)).mean():.1f} cells")


    print("\n=== SUMMARY ===")
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    print(f"Total cells: {total_cells:,}")
    print(f"Total missing cells: {total_missing:,}")
    print(f"Overall missing %: {(total_missing / total_cells) * 100:.1f}%")
    print("=" * 80 + "\n")

def get_descriptive_statistics(df, title):
    """
    Display descriptive statistics for a DataFrame.
    It includes shape, descriptive stats, head, info, and skewness.
    """
    
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()
    
    # 1. Shape
    print("=" * 80)
    print("SHAPE OF THE DATASET")
    print("=" * 80)
    print()
    print(df.shape)
    print()

    # 2. Head
    print("=" * 80)
    print("HEAD OF THE DATASET")
    print("=" * 80)
    print()
    print(df.head())
    print()
    
    # 3. Descriptive statistics
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    print()
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print(numeric_df.describe())
    print()
    
    # 4. Info
    print("=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print()
    type_summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    print(type_summary.to_string(index=False))
    print()
    
    # 5. Skewness
    print("=" * 80)
    print("SKEWNESS OF THE DATASET")
    print("=" * 80)
    print()
    print(numeric_df.skew())
    print()
    
    print("=" * 80 + "\n")

def display_correlations_with_target(df, target = "PV1MATH"):
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

    # Calculate correlations with the target and remove the target itself.
    correlations = (df.corr(method="spearman")[target].drop(target))

    # Sort positive correlations from strongest to weakest.
    positive_correlations = (correlations[correlations > 0].sort_values(ascending=False).rename("Correlation").to_frame())

    # Sort negative correlations from strongest to weakest.
    positive_correlations.index.name = "Feature"

    negative_correlations = (correlations[correlations < 0].sort_values(ascending=True).rename("Correlation").to_frame())

    negative_correlations.index.name = "Feature"

    print(f"\nPOSITIVE CORRELATIONS WITH {target}")
    print(positive_correlations)

    print(f"\nNEGATIVE CORRELATIONS WITH {target}")
    print(negative_correlations)

def display_correlations_between_features(df, target = "PV1MATH", corr_threshold = 0.70):
    """
    Displays the 20 strongest positive and negative Spearman correlations
    between features, excluding the target variable.

    It also displays feature pairs whose absolute correlation exceeds
    the specified threshold, indicating a potential risk of multicollinearity.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target (str): The target variable to exclude.
    - corr_threshold (float): The correlation threshold used to identify
      a potential risk of multicollinearity. By default, it is set to 0.70, 
      representing a relatively conservative threshold.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot calculate correlations.")
        return

    # Remove the target variable before calculating correlations.
    features = df.drop(columns=[target], errors="ignore")
    correlation_matrix = features.corr(method="spearman")
    correlations = []

    # It keeps each variable pair only once.
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

    print("\n20 STRONGEST POSITIVE CORRELATIONS BETWEEN FEATURES")
    print(positive_correlations)

    print("\n20 STRONGEST NEGATIVE CORRELATIONS BETWEEN FEATURES")
    print(negative_correlations)

    print(
        f"\nPOTENTIAL MULTICOLLINEARITY RISK "
        f"(|CORRELATION| >= {corr_threshold})"
    )

    if multicollinearity_risk.empty:
        print("No feature pair exceeds the selected corr_threshold.")
    else:
        print(multicollinearity_risk)


# ==== FEATURE SELECTION ==== 

def display_variance_threshold(ranking_VThresh: pd.DataFrame, threshold = VARIANCE_THRESHOLD, selected_cols: list = [], feature_names: list = []):
    print("\n" + "=" * 80)
    print(f"Variance Threshold (threshold = {threshold}) : {len(selected_cols)}/{len(feature_names)} kept")
    print("=" * 80)
    print()
    print(ranking_VThresh.to_string(index=False))
    print()

def display_mutual_info(ranking_MI: pd.DataFrame, task, X_train):

    print("\n" + "=" * 80)
    print(f"Mutual Information ({task}) : Complete ranking of {X_train.shape[1]} features.")
    print("=" * 80)
    print("\n Mutual Information Ranking (1 = best) :")
    print()
    print(ranking_MI.to_string(index=False))
    print()
    
def display_anova(ranking_anova: pd.DataFrame, task, X_train):
    print("\n" + "=" * 80)
    print(f"ANOVA ({task}) : Complete ranking of {X_train.shape[1]} features.")
    print("=" * 80)
    print("\n ANOVA Ranking (1 = best F-score) :")
    print(ranking_anova.to_string(index = False))

def display_rfe(ranking_rfe: pd.DataFrame, task, X_train):
    print("\n" + "=" * 80)
    print(f"RFE ({task}) : Complete ranking of {X_train.shape[1]} features.")
    print("=" * 80)
    print("\n RFE Ranking (1 = best) :")
    print(ranking_rfe.to_string(index=False))

def display_summarised_feature_rankings(combined_rankings: pd.DataFrame, method_names = None):

    print("\n" + "=" * 80)
    print("COMBINED FINAL RANKING")
    print("=" * 80)
    print(f"\n {len(combined_rankings)} combined methods: {', '.join(method_names)}")
    print("\n Final Ranking:")
    print(combined_rankings.to_string(index=False))

def display_top_features(top_features: list, k):
    print("\n" + "=" * 80)
    print(f"TOP {k} FEATURES SELECTED")
    print("=" * 80)
    print()
    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature}")


# ==== MODEL TRAINING ====

#utils
def check_overfitting_regression(train_r2, cv_score):

    # Overfitting check based on check based on the gap between Train R² and Cross-Validation R². 
    # Inspiration for the heuristic: https://datascience.stackexchange.com/questions/77298/how-many-ways-are-there-to-check-model-overfitting)
    
    gap = train_r2 - cv_score
    
    # # A 10% R² gap is a clear overfitting signal, 5% can be considered as a warning zone whiles negative gaps below -5% are rare enough to be noted as positive.
    if gap > 0.10:
        status = "Overfitting"
        detail = f"Gap (Train - CV) = {gap:.4f} (> 0.10)"
    elif gap > 0.05:
        status = "Mild overfitting"
        detail = f"Gap (Train - CV) = {gap:.4f}"
    elif gap < -0.05:
        status = "Good generalisation (Test R² > Train R²)"
        detail = f"Gap (Train - CV) = {gap:.4f}"
    else:
        status = "Good generalisation"
        detail = f"Gap (Train - CV) = {gap:.4f}"
    
    return status, detail, gap

#utils
def check_overfitting_classification(train_f1, cv_f1):

    # Overfitting check based on the gap between Train F1 and Cross-Validation F1.
    # In that situation, the threshold is stricter (0.05) than regression (0.10) because classification 
    # metrics are strictly bounded between 0 and 1. This means that in such situation, a 5% drop 
    # represents a critical loss of operational predictive power.
    gap = train_f1 - cv_f1
    
    if gap > 0.05:
        status = "Overfitting"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    elif gap > 0.02:
        status = "Mild overfitting"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    elif gap < -0.03: # A negative gap exceeding 3% is relatively rare in practice and suggests that the model generalises surprisingly well.
        status = "Good generalisation (CV > Train)"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    else:
        status = "Good generalisation"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    
    return status, detail, gap

def display_results(results, model_names=None):
    """
    Display results for all models in the results dictionary.
    """
    if model_names is None:
        model_names = list(results.keys())
    
    for name in model_names:
        if name not in results:
            print(f"Model '{name}' not found in results.")
            continue
        
        result = results[name]
        
        if result['type'] == 'regression':
            display_regression_result(name, result)
        else:
            display_classification_result(name, result)

def display_regression_result(name, result):
    """Internal function to display a single regression result."""
    print("=" * 50)
    print(f"{name} (REGRESSION) - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'R²':<12} {result['train_r2']:>10.4f} {result['test_r2']:>10.4f} {result['gap_r2']:>10.4f}")
    print(f"{'MSE':<12} {result['train_mse']:>10.2f} {result['test_mse']:>10.2f} {result['train_mse'] - result['test_mse']:>10.2f}")
    print(f"{'RMSE':<12} {result['train_rmse']:>10.4f} {result['test_rmse']:>10.4f} {result['train_rmse'] - result['test_rmse']:>10.4f}")
    print(f"{'MAE':<12} {result['train_mae']:>10.4f} {result['test_mae']:>10.4f} {result['train_mae'] - result['test_mae']:>10.4f}")
    print("-" * 50)
    
    if result.get('is_lasso'):
        print(f"Best alpha        : {result['best_alpha']}")
        print(f"Features selected : {result['n_selected']}/{result['p']}")

    if 'best_cv_score' in result:
        print(f"Best CV R²-score  : {result['best_cv_score']:.4f}")

    # Overfitting status
    cv_score = result.get('best_cv_score', result['test_r2'])
    status, detail, _ = check_overfitting_regression(result['train_r2'], cv_score)
    print(f"\n{status}: {detail}")

def display_classification_result(name, result):
    """Internal function to display a single classification result."""
    print("=" * 50)
    print(f"{name} (CLASSIFICATION) - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {result['train_accuracy']:>10.4f} {result['test_accuracy']:>10.4f} {result['gap_accuracy']:>10.4f}")
    print(f"{'F1 (macro)':<15} {result['train_f1_macro']:>10.4f} {result['test_f1_macro']:>10.4f} {result['gap_f1']:>10.4f}")
    print("-" * 50)
    
    # Classification report with original labels
    print(f"\nClassification Report {name} (Test):")
    print(classification_report(result['y_test_true'], result['y_pred']))
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(result['y_test_true'], result['y_pred']))
    print()
    print(f"Best parameters: {result['best_params']}")
    print(f"Best CV F1-score: {result['best_cv_score']:.4f}")
    
    # Overfitting status
    status, detail, _ = check_overfitting_classification(result['train_f1_macro'], result['best_cv_score'])
    print(f"\n{status}: {detail}")
    print()

def display_detailed_results_by_k(all_results, k_values=None):
    """
    Display detailed results for each k.
    """
    if k_values is None:
        k_values = sorted(all_results.keys())
    
    for k in k_values:
        print(f"\n{'#'*60}")
        print(f"# DETAILED RESULTS FOR K = {k} FEATURES")
        print(f"{'#'*60}")
        
        features = all_results[k]['features']
        display_top_features(features, k)
        
        results_k = all_results[k]['results']
        display_results(results_k)

def display_summary(results_df, task):
    """
    Display summary of metrics from results_df.
    """
    print("\n" + "=" * 80)
    print(f"SUMMARY - {task.upper()} METRICS")
    print("=" * 80)
    
    if task == 'regression':
        display_cols = ['k', 'Model', 'CV_Score', 'R²_test', 'RMSE_test', 'MAE_test', 'Gap_R²']
        if 'Features_Selected' in results_df.columns:
            display_cols.append('Features_Selected')
        print(results_df[display_cols].to_string(index=False))
    else:
        print(results_df[['k', 'Model', 'CV_Score', 'Accuracy_test', 'F1_macro_test', 'Gap_Accuracy']].to_string(index=False))

###### SHAP ANALYSIS #####

def display_global_shap_rankings(global_mean_absolute_shap_rankings_dict, top_n=10):

    for model_name, ranking_df in global_mean_absolute_shap_rankings_dict.items():
        print(f"\n{'='*60}")
        print(f"SHAP FEATURE IMPORTANCE RANKING : {model_name}, top {top_n} features")
        print(f"{'='*60}")
        print(ranking_df.head(top_n).to_string(index=False))
        print("="*60 + "\n")