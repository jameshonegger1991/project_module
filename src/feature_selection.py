import numpy as np
import pandas as pd
from sklearn.feature_selection import(
    VarianceThreshold,
    mutual_info_regression,
    mutual_info_classif,
    f_regression,
    f_classif,
    RFE,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from src.config import (
    VARIANCE_THRESHOLD,
    RANDOM_STATE,
)
from src.tables import (
    save_variance_threshold,
    save_mutual_info,
    save_anova,
    save_rfe,
    save_summarised_feature_rankings,
)

def apply_variance_threshold(X_train, feature_names, threshold = VARIANCE_THRESHOLD):
    """
    Filters out features with variance below the given threshold (fitted on training set only).

    Args:
        X_train: Training features.
        feature_names: Column names matching X_train.
        threshold: Minimum variance to keep a feature. Defaults to VARIANCE_THRESHOLD (= 0) to remove
        only constant features (conservative approach).

    Returns:
        tuple: (variance_report_df, selected_feature_names, threshold_used)
    """

    if X_train.shape[1] != len(feature_names):
        raise ValueError("feature_names must contain one name per column in X_train.")
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)

    # ================ DEBUG ===================================
    #actual_variances = np.var(X_train, axis=0)
    #comparison_df = pd.DataFrame({"Feature": feature_names, "Selector variance": selector.variances_, "Actual variance": actual_variances})
    #comparison_df["Difference"] = (comparison_df["Selector variance"]- comparison_df["Actual variance"])
    #comparison_df["Same"] = np.isclose(comparison_df["Selector variance"],comparison_df["Actual variance"])
    #print(comparison_df.to_string(index=False))
    # ==========================================================

    selected_indices = selector.get_support(indices=True)

    selected_cols = [feature_names[i] for i in selected_indices]
    
    report_df = pd.DataFrame({
        'Feature': feature_names,
        'Variance': np.var(X_train, axis=0), # np.var() is preferred over selector.variances_ as it does not internally adjust variance value when threshold = 0 (uncomment the DEBUG section above to see the comparison).
        'Status': ['KEPT' if selector.get_support()[i] else 'REMOVED' for i in range(len(feature_names))]
        }).sort_values('Variance', ascending = False
                       ).reset_index(drop=True)
    
    return report_df, selected_cols, threshold

def apply_mutual_info(X_train, y_train, feature_names, task='regression'):
    """
    Calculates feature ranking using Mutual Information scores.

    Args:
        X_train, y_train: Training features (post-variance thresholding in this pipeline) and target.
        feature_names: Column names matching X_train.
        task: 'regression' or 'classification'. Default 'regression'.

    Returns:
        pd.DataFrame: Feature ranking sorted by MI scores.
    """

    # MI captures non-linear dependencies.
    # REFERENCES:
    # - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
    # - https://medium.com/@suvendulearns/decoding-mutual-information-mi-a-guide-for-machine-learning-practitioners-b0f0ca0b30c9  

    if task == 'regression':
        mi_scores = mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE) 
    elif task == 'classification':
        mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE) 
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    ranking_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
    ranking_df['Ranking'] = (ranking_df['MI_Score'].rank(method='min', ascending=False).astype(int))
    ranking_df = ranking_df.sort_values(['Ranking', 'Feature']).reset_index(drop=True) 
    
    return ranking_df

def apply_ANOVA(X_train, y_train, feature_names, task ='regression'):
    """
    Ranks features using ANOVA F-scores (and p-values).

    Arguments:
    - X_train, y_train: Training features (post-variance thresholding in this pipeline) and target.
    - feature_names: Column names matching X_train.
    - task: 'regression' or 'classification'. Default 'regression'.

    Returns:
    - pd.DataFrame: Feature ranking with F-scores and p-values.
    """
    
    if task == 'regression':
        f_scores, p_values = f_regression(X_train, y_train)
    elif task == 'classification':
        f_scores, p_values = f_classif(X_train, y_train)
    else:
        raise ValueError("task must be 'regression' or 'classification'")
    
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': f_scores,
        'P_Value': p_values
    })
    
    ranking_df = ranking_df.sort_values('F_Score', ascending = False)
    ranking_df['Ranking'] = range(1, len(ranking_df) + 1)
    
    return ranking_df

def apply_rfe(X_train, y_train, feature_names, task ='regression', step = 1):
    """
    Ranks features using Recursive Feature Elimination (RFE) with a Random Forest estimator.

    Arguments:
    - X_train, y_train: Training features (post-variance thresholding in this pipeline) and target.
    - feature_names: Column names matching X_train.
    - task: 'regression' or 'classification'. Default set to 'regression'.
    - step: Number of features removed at each iteration. Default 1.

    Returns:
    - pd.DataFrame: Complete feature ranking from RFE.
    """
    
    if task == 'regression':
        estimator = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    elif task == 'classification':
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    else:
        raise ValueError("task must be 'regression' or 'classification'")
    
    # RFE must eliminate down to one feature to generate a complete distinct ranking because stopping earlier would leave all remaining features at rank 1.
    selector = RFE(estimator = estimator, n_features_to_select = 1, step = step)
    selector.fit(X_train, y_train)
    
    rfe_df = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': selector.ranking_
    }).sort_values('Ranking')
    
    return rfe_df

def summarise_feature_rankings(ranking_dfs, method_names = None):
    """
    Aggregates multiple feature rankings (from MI, ANOVA, RFE, etc.) into a final ranking using rank sum.

    Arguments:
    - ranking_dfs: List of DataFrames, each with feature rankings from a different method.
    - method_names: Names for each method. Defaults to auto-generated names.

    Returns:
    - tuple: (combined_ranking_df, method_names_list)
    """
    
    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(len(ranking_dfs))]
    
    combined_df = ranking_dfs[0][['Feature', 'Ranking']].copy()
    combined_df = combined_df.rename(columns={'Ranking': method_names[0]})
    
    for i in range(1, len(ranking_dfs)):
        df = ranking_dfs[i][['Feature', 'Ranking']].copy()
        df = df.rename(columns={'Ranking': method_names[i]})
        combined_df = combined_df.merge(df, on='Feature', how='outer')
    
    combined_df['Sum of ranks'] = combined_df[method_names].sum(axis=1)
    
    # Ex-aequo features get the minimum rank.
    # REFERENCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html 
    combined_df['Final ranking'] = combined_df['Sum of ranks'].rank(method='min').astype(int)
    combined_df = combined_df.sort_values('Final ranking')
    
    return combined_df, method_names

def select_top_features(combined_rankings, k = 20):
    """
    Extracts the top k features from the aggregated ranking (output of summarise_feature_rankings())
    and returns a tuple with the selected features (list) and the number of features selected (k).

    Arguments:
    - combined_rankings: Final ranking DataFrame.
    - k: Number of top features to keep. Default 20.
    """
    
    if k > len(combined_rankings):
        k = len(combined_rankings)
    
    if k <= 0 :
        raise ValueError("k must be greater than 0")
    
    top_features = combined_rankings.head(k)['Feature'].tolist()
    
    return top_features, k

def run_feature_selection_pipeline(X_train, X_test, y_train, feature_names, var_threshold = VARIANCE_THRESHOLD, task: str = 'regression'):
    """
    Executes the whole feature selection process and filters X_train/X_test with top features.
    """
    variance_threshold_df, selected_columns_var_thresh, threshold = apply_variance_threshold(X_train, feature_names, threshold = var_threshold)
    save_variance_threshold(variance_threshold_df, threshold, selected_columns_var_thresh, feature_names)

    selected_indices = [feature_names.index(col) for col in selected_columns_var_thresh]
    X_train_after_var_thresh = X_train[:, selected_indices]
    X_test_after_var_thresh = X_test[:, selected_indices]
    
    ranking_MI_df = apply_mutual_info(X_train_after_var_thresh, y_train, selected_columns_var_thresh, task = task)
    save_mutual_info(ranking_MI_df, task, X_train_after_var_thresh)

    ranking_anova_df = apply_ANOVA(X_train_after_var_thresh, y_train, selected_columns_var_thresh, task = task)
    save_anova(ranking_anova_df, task, X_train_after_var_thresh)

    ranking_rfe_df= apply_rfe(X_train_after_var_thresh, y_train, selected_columns_var_thresh, task = task, step=1)
    save_rfe(ranking_rfe_df, task, X_train_after_var_thresh)

    final_features_ranking, method_names = summarise_feature_rankings([ranking_MI_df, ranking_anova_df, ranking_rfe_df], ["Mutual Information", "ANOVA", "RFE"])
    save_summarised_feature_rankings(final_features_ranking, method_names, task)

    #top_features, k = select_top_features(final_features_ranking, 10)
    #save_top_features(top_features, k)

    return final_features_ranking, variance_threshold_df, ranking_MI_df, ranking_anova_df, ranking_rfe_df, X_train_after_var_thresh, X_test_after_var_thresh, selected_columns_var_thresh