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

def apply_variance_threshold(X_train, feature_names, threshold = 0.01):
    """
    Applies variance thresholding to select features and generates a report and returns a tuple
    containing the report DataFrame and a list of selected feature names.
    """
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(X_train)
    selected_indices = selector.get_support(indices=True) 
    selected_cols = [feature_names[i] for i in selected_indices]
    
    variances = np.var(X_train, axis=0)
    report_df = pd.DataFrame({
        'Feature': feature_names,
        'Variance': variances,
        'Status': [' KEPT' if selector.get_support()[i] else 'REMOVED' for i in range(len(feature_names))]
    }).sort_values('Variance', ascending = False) # To display the most variable features first.
    
    print(f"\n Variance Threshold (threshold = {threshold}) : {len(selected_cols)}/{len(feature_names)} kept")
    print()
    print(report_df.to_string(index=False))
    
    return report_df, selected_cols


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
    else:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=7)
    
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': mi_scores,
        'Ranking': mi_scores.argsort()[::-1] + 1 # Creates a 1-based ranking where the feature with the highest MI score gets rank 1.
    }).sort_values('Ranking')
    
    print(f" Mutual Information ({task}) : Classement complet de {X_train.shape[1]} features.")
    print("\n Classement Mutual Information (1 = meilleur) :")
    print(ranking_df.to_string(index=False))
    
    return ranking_df

def apply_ANOVA(X_train, y_train, feature_names, task='regression'):
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
    ranking_df = ranking_df.sort_values('F_Score', ascending=False)
    
    # Add the ranking (1 = best F-score)
    ranking_df['Ranking'] = range(1, len(ranking_df) + 1)
    
    print(f" ANOVA ({task}) : Complete ranking of {X_train.shape[1]} features.")
    print("\n ANOVA Ranking (1 = best F-score) :")
    print(ranking_df.to_string(index=False))
    
    return ranking_df


def apply_rfe(X_train, y_train, feature_names, task='regression', step = 1):
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
    
    print(f" RFE ({task}) : Complete ranking of {X_train.shape[1]} features.")
    print("\n RFE Ranking (1 = best) :")
    print(rfe_df.to_string(index=False))
    
    return rfe_df

def summarise_feature_rankings(ranking_dfs, method_names=None):
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
    
    print("\n" + "=" * 80)
    print("COMBINED FINAL RANKING")
    print("=" * 80)
    print(f"\n {len(ranking_dfs)} combined methods: {', '.join(method_names)}")
    print("\n Final Ranking:")
    print(combined_df.to_string(index=False))
    
    return combined_df

def select_top_features(combined_rankings, k = 20):
    """
    Picks the top 'k' features from the combined ranking and returns a list of their names.
    """
    
    if k > len(combined_rankings):
        k = len(combined_rankings)
    
    if k <= 0 :
        raise ValueError("k must be greater than 0")
    
    top_features = combined_rankings.head(k)['Feature'].tolist()
    
    print("\n" + "=" * 80)
    print(f"TOP {k} FEATURES SELECTED")
    print("=" * 80)
    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature}")
    
    return top_features

def run_feature_selection_pipeline(X_train, X_test, y_train, feature_names):
    variance_treshold_df, selected_columns_var_thresh = apply_variance_threshold(X_train, feature_names, threshold = 0)
    selected_indices = [feature_names.index(col) for col in selected_columns_var_thresh]
    X_train_after_var_thresh = X_train[:, selected_indices]
    X_test_after_var_thresh = X_test[:, selected_indices]
    
    ranking_MI_df = apply_mutual_info(X_train_after_var_thresh, y_train, feature_names, task='regression')
    ranking_anova_df = apply_ANOVA(X_train_after_var_thresh, y_train, feature_names, task='regression')
    ranking_rfe_df = apply_rfe(X_train_after_var_thresh, y_train, feature_names, task='regression', step=1)
    final_features_ranking = summarise_feature_rankings([ranking_MI_df, ranking_anova_df, ranking_rfe_df], ["Mutual Information", "ANOVA", "RFE"])
    select_top_features(final_features_ranking, 10)

    return final_features_ranking, variance_treshold_df, ranking_MI_df, ranking_anova_df, ranking_rfe_df, X_train_after_var_thresh, X_test_after_var_thresh