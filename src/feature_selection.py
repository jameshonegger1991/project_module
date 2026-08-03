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
)
from src.tables import (
    display_variance_threshold,
    display_mutual_info,
    display_anova,
    display_rfe,
    display_summarised_feature_rankings,
    display_top_features,
)

def apply_variance_threshold(X_train, feature_names, threshold = VARIANCE_THRESHOLD):
    """
    Applies variance thresholding to select features and generates a report and returns a tuple
    containing the report DataFrame and a list of selected feature names.
    """

    if X_train.shape[1] != len(feature_names):
        raise ValueError("feature_names must contain one name per column in X_train.")
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)

    selected_indices = selector.get_support(indices=True)

    selected_cols = [feature_names[i] for i in selected_indices]
    
    report_df = pd.DataFrame({
        'Feature': feature_names,
        'Variance': selector.variances_,
        'Status': ['KEPT' if selector.get_support()[i] else 'REMOVED' for i in range(len(feature_names))]
        }).sort_values('Variance', ascending = False
                       ).reset_index(drop=True)
    
    return report_df, selected_cols, threshold

def apply_mutual_info(X_train, y_train, feature_names, task='regression'):
    """
    Returns the Mutual Information (MI) ranking of features in a DataFrame. 
    """

    # modular feature selection test able to work either with regression or classification tasks.
    # Mutual information test has the advantage to be model-independent. A higher MI score indicates a stronger dependency between 
    # the feature and the target variable.
    # It quantifies the amount of information obtained about one random variable by observing the other.
    # REFERENCES:
    # - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
    # - https://medium.com/@suvendulearns/decoding-mutual-information-mi-a-guide-for-machine-learning-practitioners-b0f0ca0b30c9  

    if task == 'regression':
        mi_scores = mutual_info_regression(X_train, y_train, random_state=7) 
    elif task == 'classification':
        mi_scores = mutual_info_classif(X_train, y_train, random_state=7) 
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    ranking_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
    ranking_df['Ranking'] = (ranking_df['MI_Score'].rank(method='min', ascending=False).astype(int)) # Ranks features based on MI scores, assigning the same rank to tied scores.
    ranking_df = ranking_df.sort_values(['Ranking', 'Feature']).reset_index(drop=True) # Sorts the DataFrame by rank and then by feature name, resetting the index.
    
    return ranking_df

def apply_ANOVA(X_train, y_train, feature_names, task ='regression'):
    """
    Applies the ANOVA F-test to features and returns a complete ranking (df) with feature names, F-scores, p-values, and their ranking.
    """
    
    if task == 'regression':
        f_scores, p_values = f_regression(X_train, y_train)
    elif task == 'classification':
        f_scores, p_values = f_classif(X_train, y_train)
    else:
        raise ValueError("task must be 'regression' or 'classification'")
    
    # Créer le DataFrame avec les scores
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': f_scores,
        'P_Value': p_values
    })
    
    # Sort by F-Score in descending order (from largest to smallest)
    ranking_df = ranking_df.sort_values('F_Score', ascending = False)
    
    # Add the ranking (1 = best F-score)
    ranking_df['Ranking'] = range(1, len(ranking_df) + 1)
    
    return ranking_df

def apply_rfe(X_train, y_train, feature_names, task ='regression', step = 1):
    """
    Performs Recursive Feature Elimination (RFE) with a RandomForest estimator to return a complete feature ranking with feature names. 
    """
    
    if task == 'regression':
        estimator = RandomForestRegressor(n_estimators=100, random_state=7)
    elif task == 'classification':
        estimator = RandomForestClassifier(n_estimators=100, random_state=7)
    else:
        raise ValueError("task must be 'regression' or 'classification'")
    
    # n_features_to_select = 1 to obtain a complete ranking.
    # If n_features_to_select were greater than 1, RFE would stop once that number of features was reached,
    # and all remaining (non-eliminated) features would have the same rank (1),
    # thus not providing a distinct ranking for all features.
    selector = RFE(estimator = estimator, n_features_to_select = 1, step = step)
    selector.fit(X_train, y_train)
    
    rfe_df = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': selector.ranking_
    }).sort_values('Ranking')
    
    return rfe_df

def summarise_feature_rankings(ranking_dfs, method_names = None):
    """
    Combines multiple feature rankings into a single final ranking and returns it as a DataFrame that contains:
    - features
    - their individual rankings from each method
    - the sum of their ranks
    - a final combined ranking
    """
    
    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(len(ranking_dfs))]
    
    combined_df = ranking_dfs[0][['Feature', 'Ranking']].copy()
    combined_df = combined_df.rename(columns={'Ranking': method_names[0]})
    
    for i in range(1, len(ranking_dfs)):
        df = ranking_dfs[i][['Feature', 'Ranking']].copy()
        df = df.rename(columns={'Ranking': method_names[i]})
        # Merge on 'Feature' to correctly align the rankings.
        combined_df = combined_df.merge(df, on='Feature', how='outer')
    
    combined_df['Sum of ranks'] = combined_df[method_names].sum(axis=1)
    
    # Calculates the final ranking by assigning a rank (1 = best) based on the sum of ranks. The `.rank()` method converts
    # the numerical 'Sum of ranks' into an ordered ranking.
    # The method='min' is chosen for the following reason: if multiple features have the same 'Sum of ranks',
    # they will all be assigned the "lowest" rank that any of them would have received. For example, if two features
    # are tied for what would be the 2nd and 3rd positions, both will receive rank 2. This ensures a consistent
    # and unambiguous ranking for all tied values and gives the assurance that no significative feature is wrongly ranked.
    # REFERENCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html 
    combined_df['Final ranking'] = combined_df['Sum of ranks'].rank(method='min').astype(int)
    combined_df = combined_df.sort_values('Final ranking')
    
    return combined_df, method_names

def select_top_features(combined_rankings, k = 20):
    """
    Picks the top 'k' features from the combined ranking and returns a list of their names.
    """
    
    if k > len(combined_rankings):
        k = len(combined_rankings)
    
    if k <= 0 :
        raise ValueError("k must be greater than 0")
    
    top_features = combined_rankings.head(k)['Feature'].tolist()
    
    return top_features, k

def run_feature_selection_pipeline(X_train, X_test, y_train, feature_names, var_threshold = VARIANCE_THRESHOLD, task: str = 'regression'):

    variance_treshold_df, selected_columns_var_thresh, threshold = apply_variance_threshold(X_train, feature_names, threshold = var_threshold)
    display_variance_threshold(variance_treshold_df, threshold, selected_columns_var_thresh, feature_names)

    selected_indices = [feature_names.index(col) for col in selected_columns_var_thresh]
    X_train_after_var_thresh = X_train[:, selected_indices]
    X_test_after_var_thresh = X_test[:, selected_indices]
    
    ranking_MI_df = apply_mutual_info(X_train_after_var_thresh, y_train, selected_columns_var_thresh, task = task)
    display_mutual_info(ranking_MI_df, task, X_train_after_var_thresh)

    ranking_anova_df = apply_ANOVA(X_train_after_var_thresh, y_train, selected_columns_var_thresh, task = task)
    display_anova(ranking_anova_df, task, X_train_after_var_thresh)

    ranking_rfe_df= apply_rfe(X_train_after_var_thresh, y_train, selected_columns_var_thresh, task = task, step=1)
    display_rfe(ranking_rfe_df, task, X_train_after_var_thresh)

    final_features_ranking, method_names = summarise_feature_rankings([ranking_MI_df, ranking_anova_df, ranking_rfe_df], ["Mutual Information", "ANOVA", "RFE"])
    display_summarised_feature_rankings(final_features_ranking, method_names)

    #top_features, k = select_top_features(final_features_ranking, 10)
    #display_top_features(top_features, k)

    return final_features_ranking, variance_treshold_df, ranking_MI_df, ranking_anova_df, ranking_rfe_df, X_train_after_var_thresh, X_test_after_var_thresh