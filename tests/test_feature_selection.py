import pytest
import numpy as np
import pandas as pd
from src.feature_selection import (
    apply_variance_threshold,
    apply_mutual_info,
    apply_ANOVA,
    apply_rfe,
    summarise_feature_rankings,
    select_top_features
)
from sklearn.datasets import make_regression, make_classification

def test_apply_zero_variance_threshold():
    """
    The main goal here is to verify that the function correctly removes features
    with zero variance when the threshold is set to 0.
    This is a fundamental step in feature selection to eliminate uninformative columns.
    """
    
    # Setting up a simple dataset. Feature 'B' is constant (variance = 0),
    # while 'A' and 'C' have varying values.
    X = np.array([[1, 2, 3], [2, 2, 4], [3, 2, 5]])
    feature_names = ['A', 'B', 'C']
    
    # Calling the function with `threshold=0`.
    # Based on `sklearn.feature_selection.VarianceThreshold` behavior,
    # features with variance *equal to* 0 should be removed.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    report_df, selected_cols, threshold_returned, feature_names_returned = apply_variance_threshold(
        X, feature_names, threshold=0
    )
    
    # Assertions to confirm the expected outcome:
    # - Only 'A' and 'C' should be kept.
    # - 'B' should be explicitly removed.
    assert len(selected_cols) == 2
    assert 'B' not in selected_cols
    assert 'A' in selected_cols
    assert 'C' in selected_cols
    
    # Also, checking if the report DataFrame is structured as expected and reflects the status correctly.
    assert 'Feature' in report_df.columns
    assert 'Variance' in report_df.columns
    assert 'Status' in report_df.columns
    assert report_df[report_df['Feature'] == 'B']['Status'].iloc[0] == 'REMOVED'
    assert report_df[report_df['Feature'] == 'A']['Status'].iloc[0] == ' KEPT'
    assert report_df[report_df['Feature'] == 'C']['Status'].iloc[0] == ' KEPT'

    # And verifying that the returned metadata matches the input.
    assert threshold_returned == 0
    assert feature_names_returned == feature_names

def test_apply_variance_threshold_with_higher_threshold():
    """
    The threshold is set to 0.5, which should remove features with variance < 0.5.
    In our dataset, feature 'A' has variance around 0.67, 'B' has 0, and 'C' has  around 0.67.
    Only 'B' should be removed (variance 0 < 0.5).
    'A' and 'C' have variance > 0.5, so they should be kept.
    """
    X = np.array([[1, 2, 3], [2, 2, 4], [3, 2, 5]])
    feature_names = ['A', 'B', 'C']
    
    report_df, selected_cols, threshold_returned, feature_names_returned = apply_variance_threshold(
        X, feature_names, threshold=0.5
    )
    
    # Only 'B' should be removed (variance = 0 < 0.5)
    # 'A' and 'C' have variance ~0.67 > 0.5, so they stay
    assert len(selected_cols) == 2
    assert 'B' not in selected_cols
    assert 'A' in selected_cols
    assert 'C' in selected_cols
    assert threshold_returned == 0.5
    assert feature_names_returned == feature_names
    
    assert report_df[report_df['Feature'] == 'B']['Status'].iloc[0] == 'REMOVED'
    assert report_df[report_df['Feature'] == 'A']['Status'].iloc[0] == ' KEPT'
    assert report_df[report_df['Feature'] == 'C']['Status'].iloc[0] == ' KEPT'

def test_apply_mutual_info():
    """
    This test doesn't check the exact order of features, but rather that the ranking
    is consistent with the data generation. This test ensures that the "apply_mutual_info"
    function correctly identifies informative features for both regression and classification tasks.
    """
    
    # --- Regression ---
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
    X_reg, y_reg, coef = make_regression(
        n_samples=500,
        n_features=3,
        n_informative=2,
        noise=1.0,
        random_state=42,
        coef=True, # Return the coefficients to know which features are informative.
        shuffle=False # Keep informative features (A, B) at the start for easier checking.
    )
    feature_names = ['A', 'B', 'C']
    
    ranking_df_reg = apply_mutual_info(X_reg, y_reg, feature_names, task='regression')
    
    top_2 = ranking_df_reg.head(2)['Feature'].tolist()
    assert 'A' in top_2
    assert 'B' in top_2
    assert ranking_df_reg.iloc[2]['Feature'] == 'C'  # 'C' should be the last (least informative) feature.
    
    # --- Classification ---
    # Generate a synthetic classification dataset. Similar parameters to make_regression.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    X_clf, y_clf = make_classification(
        n_samples=500,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=42,
        shuffle=False
    )
    feature_names = ['A', 'B', 'C']
    
    # Apply mutual information for classification.
    ranking_df_clf = apply_mutual_info(X_clf, y_clf, feature_names, task='classification')
    
    top_2_clf = ranking_df_clf.head(2)['Feature'].tolist()
    assert 'A' in top_2_clf
    assert 'B' in top_2_clf
    assert ranking_df_clf.iloc[2]['Feature'] == 'C'

def test_apply_anova():
    """
    For regression, ANOVA (F-test) detects linear relationships between features and
    a continuous target; and for classification, ANOVA detects differences in feature means across classes.
    """
    
    # Part 1: Regression task
    # Generating a dataset where Feature A has a strong linear relationship with y.
    # ANOVA (F-test) is good for detecting linear dependencies.
    X_reg = np.random.rand(100, 3)
    y_reg = X_reg[:, 0] * 2 + np.random.randn(100) * 0.1 #only Feature A has a impact on the target
    feature_names = ['A', 'B', 'C']
    
    ranking_df_reg = apply_ANOVA(X_reg, y_reg, feature_names, task='regression')
    
    assert 'Feature' in ranking_df_reg.columns
    assert 'F_Score' in ranking_df_reg.columns
    assert 'P_Value' in ranking_df_reg.columns
    assert 'Ranking' in ranking_df_reg.columns

    assert ranking_df_reg.iloc[0]['Feature'] == 'A'
    assert ranking_df_reg.iloc[0]['Ranking'] == 1
    assert ranking_df_reg['Ranking'].is_unique
    assert len(ranking_df_reg) == len(feature_names)
    
    # Part 2: Classification task 

    # Generating a dataset where Feature A has different means across classes.
    # For classification, ANOVA detects if the mean of a feature differs between classes.
    X_clf = np.random.rand(100, 3)

    # y_clf depends on X[:, 0]: if A > 0.5, class 1; else class 0
    y_clf = (X_clf[:, 0] > 0.5).astype(int)

    feature_names = ['A', 'B', 'C']
    
    ranking_df_clf = apply_ANOVA(X_clf, y_clf, feature_names, task='classification')
    
    assert 'Feature' in ranking_df_clf.columns
    assert 'F_Score' in ranking_df_clf.columns
    assert 'P_Value' in ranking_df_clf.columns
    assert 'Ranking' in ranking_df_clf.columns

    assert ranking_df_clf.iloc[0]['Feature'] == 'A'
    assert ranking_df_clf.iloc[0]['Ranking'] == 1
    assert ranking_df_clf['Ranking'].is_unique
    assert len(ranking_df_clf) == len(feature_names)

def test_apply_rfe():
    """
    Testing apply_rfe for both regression and classification tasks.
    """
    
    feature_names = ['A', 'B', 'C', 'D', 'E']
    
    # Part 1: Regression task
    X_reg, y_reg = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=1, # for easy checking
        noise=0.1,
        random_state=42,
        shuffle=False # to keep the same order and have a reproductible test
    )
    
    ranking_df_reg = apply_rfe(X_reg, y_reg, feature_names, task='regression', step=1)
    
    assert 'Feature' in ranking_df_reg.columns
    assert 'Ranking' in ranking_df_reg.columns
    assert ranking_df_reg.iloc[0]['Feature'] == 'A'
    assert ranking_df_reg.iloc[0]['Ranking'] == 1
    assert ranking_df_reg['Ranking'].max() == len(feature_names)
    assert ranking_df_reg['Ranking'].is_unique
    assert len(ranking_df_reg) == len(feature_names)
    
    # Part 2: Classification task
    # For n_informative = 1, n_clusters_per_class should be 1.
    X_clf, y_clf = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=1,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
        shuffle=False # Keep informative features at the start for easier checking.
    )
    
    ranking_df_clf = apply_rfe(X_clf, y_clf, feature_names, task='classification', step=1)
    
    assert 'Feature' in ranking_df_clf.columns
    assert 'Ranking' in ranking_df_clf.columns
    assert ranking_df_clf.iloc[0]['Feature'] == 'A'
    assert ranking_df_clf.iloc[0]['Ranking'] == 1
    assert ranking_df_clf['Ranking'].max() == len(feature_names)
    assert ranking_df_clf['Ranking'].is_unique
    assert len(ranking_df_clf) == len(feature_names)

def test_summarise_feature_rankings():
    """
    Testing that summarise_feature_rankings correctly combines multiple
    individual feature rankings into a single, coherent final ranking.
    """
    
    # Some dummy ranking DataFrames is created from different methods.
    # Some ties and different orderings are introduced to test robustness.
    mi_ranking = pd.DataFrame({
        'Feature': ['F1', 'F2', 'F3', 'F4'],
        'Ranking': [1, 2, 3, 4],
        })

    anova_ranking = pd.DataFrame({
        'Feature': ['F2', 'F1', 'F3', 'F4'],
        'Ranking': [1, 2, 3, 4],
        })

    rfe_ranking = pd.DataFrame({
        'Feature': ['F1', 'F4', 'F2', 'F3'],
        'Ranking': [1, 2, 3, 4],
        })
    
    ranking_dfs = [mi_ranking, anova_ranking, rfe_ranking]
    method_names = ["MI", "ANOVA", "RFE"]
    
    combined_df, returned_method_names = summarise_feature_rankings(ranking_dfs, method_names)
    
    # Assertions:
    # - Check for expected columns in the combined DataFrame.
    # - Verify the 'Sum of ranks' calculation.
    # - Ensure 'Final ranking' is correctly computed and sorted.
    assert 'Feature' in combined_df.columns
    assert 'MI' in combined_df.columns
    assert 'ANOVA' in combined_df.columns
    assert 'RFE' in combined_df.columns
    assert 'Sum of ranks' in combined_df.columns
    assert 'Final ranking' in combined_df.columns
    
    # Check sums for specific features
    assert combined_df[combined_df['Feature'] == 'F1']['Sum of ranks'].iloc[0] == 4
    assert combined_df[combined_df['Feature'] == 'F2']['Sum of ranks'].iloc[0] == 6
    assert combined_df[combined_df['Feature'] == 'F3']['Sum of ranks'].iloc[0] == 10
    assert combined_df[combined_df['Feature'] == 'F4']['Sum of ranks'].iloc[0] == 10
    
    # Check final ranking (should be sorted by 'Sum of ranks' ascending)
    assert combined_df.iloc[0]['Feature'] == 'F1'
    assert combined_df.iloc[0]['Final ranking'] == 1
    assert combined_df.iloc[1]['Feature'] == 'F2'
    assert combined_df.iloc[1]['Final ranking'] == 2
    assert combined_df.iloc[2]['Feature'] == 'F3'
    assert combined_df.iloc[2]['Final ranking'] == 3
    assert combined_df.iloc[3]['Feature'] == 'F4'
    assert combined_df.iloc[3]['Final ranking'] == 3
    
    assert returned_method_names == method_names

def test_select_top_features():
    """
    Testing that select_top_features correctly extracts the 'k' highest-ranked features
    from a combined ranking DataFrame.
    """
    
    # Creating a sample combined ranking DataFrame.
    combined_ranking_df = pd.DataFrame({
        'Feature': ['F1', 'F2', 'F3', 'F4', 'F5'],
        'MI': [1, 2, 3, 4, 5],
        'ANOVA': [1, 2, 3, 4, 5],
        'RFE': [1, 2, 3, 4, 5],
        'Sum of ranks': [3, 6, 9, 12, 15],
        'Final ranking': [1, 2, 3, 4, 5]
    }).sort_values('Final ranking') # Ensure it's sorted as expected from "summarise_feature_rankings"
    
    # Test case 1: Select a normal number of top features (e.g., k = 3)
    top_features_k3, k_returned = select_top_features(combined_ranking_df, k=3)
    assert top_features_k3 == ['F1', 'F2', 'F3']
    assert k_returned == 3
    
    # Test case 2: Select all features (k > total features)
    top_features_all, k_returned_all = select_top_features(combined_ranking_df, k = 10)
    assert top_features_all == ['F1', 'F2', 'F3', 'F4', 'F5']
    assert k_returned_all == 5 # k should be adjusted to the total number of features
    
    # Test case 3: Select k = 1
    top_features_k1, k_returned_k1 = select_top_features(combined_ranking_df, k=1)
    assert top_features_k1 == ['F1']
    assert k_returned_k1 == 1
    
    # Test case 4: k = 0 should raise a ValueError
    with pytest.raises(ValueError, match="k must be greater than 0"):
        select_top_features(combined_ranking_df, k=0)
    
    # Test case 5: k negative should also raise a ValueError
    with pytest.raises(ValueError, match="k must be greater than 0"):
        select_top_features(combined_ranking_df, k=-2)
