from sklearn.datasets import make_regression, make_classification
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from src.model_training import run_models, evaluate_k_values
import warnings



def test_run_models_regression():
    """
    Integration test to ensure the regression pipeline runs without errors 
    using a tiny synthetic dataset.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*fork.*")

    X, y = make_regression(n_samples=100, n_features=10, random_state=7)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    results = run_models(
        X_train, y_train, X_test, y_test, 
        model_names=['LR','RF','XGBoost'],
        task='regression'
    )
    
    assert ['LR','RF','XGBoost'] == list(results.keys())
    assert 'SVR' not in results
    assert results['LR']['type'] == 'regression'
    assert results['RF']['type'] == 'regression'
    assert results['XGBoost']['type'] == 'regression'
    assert len(results['LR']['y_pred']) == len(y_test)

def test_run_models_classification():
    """
    Integration test to ensure the classification pipeline runs without errors 
    using a tiny synthetic dataset.
    """
    # Ignore warnings related to synthetic data and small sample sizes, specifically strong in classification task.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*fork.*")

    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=7)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    results = run_models(
        X_train, y_train, X_test, y_test, 
        model_names=['LR','XGBoost','SVC'],
        task='classification'
    )
    
    assert ['LR','XGBoost','SVC'] == list(results.keys())
    assert 'RF' not in results
    assert results['LR']['type'] == 'classification'
    assert results['XGBoost']['type'] == 'classification'
    assert results['SVC']['type'] == 'classification'
    assert len(results['LR']['y_pred']) == len(y_test)

def test_evaluate_k_values_regression():
    """
    Integration test for evaluate_k_values to verify DataFrame and dictionary aggregation across multiple k (for regression task)
    """
    X, y = make_regression(n_samples=100, n_features=5, random_state=7)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    final_features_ranking = pd.DataFrame({
        'Feature': ['MMINS', 'PAREDINT', 'IMMIG', 'GRADE','MATHEFF'],
        'Final ranking': [1, 2, 3, 4, 5]
    })
    feature_names = ['MMINS', 'PAREDINT', 'IMMIG', 'GRADE','MATHEFF']
    
    results_df, all_results = evaluate_k_values(
        X_train, y_train, X_test, y_test,
        final_features_ranking = final_features_ranking,
        feature_names = feature_names,
        model_names = ['LR','RF','XGBoost'],
        task='regression',
        k_values=[2, 4]
    )
    
    assert isinstance(results_df, pd.DataFrame)
    assert 'k' in results_df.columns
    assert 'Model' in results_df.columns
    assert 2 in all_results
    assert 4 in all_results
    assert len(results_df) == 6  #(3 model * 2 k values)


def test_evaluate_k_values_classification():
    """
    Integration test for evaluate_k_values to verify DataFrame and dictionary aggregation across multiple k (for classification task)
    """
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    X, y = make_classification(n_samples = 100, n_features = 5, n_classes = 3, n_clusters_per_class = 1, random_state = 7)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    final_features_ranking = pd.DataFrame({
        'Feature': ['MMINS', 'PAREDINT', 'IMMIG', 'GRADE','MATHEFF'],
        'Final ranking': [1, 2, 3, 4, 5]
    })
    feature_names = ['MMINS', 'PAREDINT', 'IMMIG', 'GRADE','MATHEFF']
    
    results_df, all_results = evaluate_k_values(
        X_train, y_train, X_test, y_test,
        final_features_ranking = final_features_ranking,
        feature_names = feature_names,
        model_names = ['LR','RF','XGBoost','SVC'],
        task='classification',
        k_values=[1, 3]
    )
    
    assert isinstance(results_df, pd.DataFrame)
    assert 'k' in results_df.columns
    assert 'Model' in results_df.columns
    assert 1 in all_results
    assert 3 in all_results
    assert len(results_df) == 8  #(4 model * 2 k values)


def test_run_models_invalid_task():
    """
    Ensure that passing an unsupported task returns an empty dictionary whithout errors.
    """
    X, y = make_regression(n_samples=10, n_features=2, random_state=7)
    results = run_models(X, y, X, y, model_names=['LR'], task='clustering')
    
    assert results == {}
