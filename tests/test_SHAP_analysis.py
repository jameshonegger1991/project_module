import numpy as np
import pandas as pd
import pytest
from src.SHAP_analysis import(
    assess_features_robustness,
    feature_agreement_stats, 
    inter_model_concordance_assessment, 
    intra_model_stability_assessment, 
    rashomon_set_builder,
)

def test_rashomon_set_builder_selects_models_accurately():

    results = {
        10: {
            'results': {
                'LR': {'best_cv_score': 0.44},
                'RF': {'best_cv_score': 0.49},
                'XGBoost': {'best_cv_score': 0.40},
                'SVR': {'best_cv_score': 0.47},
                }
            }
        }

    rashomon_set, best_score, lowest_score = rashomon_set_builder(results, 10, task='regression', rashomon_threshold=0.05)

    assert round(best_score, 2) == 0.49
    assert round(lowest_score, 2) == 0.44 # lowest score = best score - rashomon_threshold
    assert set(rashomon_set.keys()) == {'LR', 'RF', 'SVR'}

def test_rashomon_set_builder_raises_error_accurately():

    with pytest.raises(ValueError, match="model_training_all_results_dic must be a valid dictionary"):
        rashomon_set_builder(None, 10)

    with pytest.raises(ValueError, match="Invalid task. Task must be 'regression' or 'classification'"):
        rashomon_set_builder({}, 10, task='unknown')

    with pytest.raises(ValueError, match="rashomon_threshold must be non-negative"):
        rashomon_set_builder({}, 10, task='regression', rashomon_threshold=-0.1)

    with pytest.raises(ValueError, match="No results found for k = 5 in the dictionary"):
        rashomon_set_builder({}, 5, task='regression', rashomon_threshold=0.05)

    with pytest.raises(ValueError, match="No CV scores found for k = 5."):
        rashomon_set_builder({5: {'results': {}}}, 5, task='regression', rashomon_threshold=0.05)
    
def test_inter_model_concordance_assessment():

    rankings = {
        'LR': pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Mean_Abs_SHAP': [3.0, 1.0, 2.0],
            'SHAP_Ranking': [1, 3, 2]
            }),
        'RF': pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Mean_Abs_SHAP': [4.0, 3.0, 2.0],
            'SHAP_Ranking': [1, 2, 3]
            })
        }

    result = inter_model_concordance_assessment(rankings, top_k_features_concordance=2)

    row = result.iloc[0]
    assert row['Kendall_coeff'] == pytest.approx(0.3333333)
    assert row['Spearman coeff'] == pytest.approx(0.5)
    assert row['Overlap ratio'] == pytest.approx(0.5) #only feature A is kept.

def test_feature_agreement_stats():

    rankings = {
        'LR': pd.DataFrame({
            'Feature': ['A', 'C', 'B'],
            'Mean_Abs_SHAP': [3.0, 2.0, 1.0],
            'SHAP_Ranking': [1, 2, 3]
        }),
        'RF': pd.DataFrame({
            'Feature': ['B', 'A', 'C'],
            'Mean_Abs_SHAP': [3.0, 2.0, 1.0],
            'SHAP_Ranking': [1, 2, 3]
        })
    }

    result = feature_agreement_stats(rankings)
    first_row = result.iloc[0]

    #Expected ranking: A: 1, B: 2, C: 3
    assert first_row['LR SHAP RANKING'] == 1
    assert first_row['RF SHAP RANKING'] == 2
    assert first_row['Mean SHAP rank'] == pytest.approx(1.5)
    assert first_row['Standard deviation SHAP rank'] == pytest.approx(np.std([1, 2], ddof=1)) # by default, pandas.std() uses ddof=1.

def test_intra_model_stability_assessment():

    shap_values = np.array([
        [1.0, 2.0, 6.0],
        [2.0, 1.0, 2.0],
        [3.0, 4.0, 2.0],
        [4.0, 3.0, 4.0],
    ])
    
    local_shap_dict = {
        'LR': {
            'shap_values': shap_values,
            'X_test_subset': np.zeros((4, 3))
            }
        }
    
    result_1 = intra_model_stability_assessment(
        local_shap_dict,
        ['A', 'B', 'C'],
        n_iterations = 50,
        random_seed = 7
    )
    
    result_2 = intra_model_stability_assessment(
        local_shap_dict,
        ['A', 'B', 'C'],
        n_iterations = 50,
        random_seed = 7  
    )

    #check if the function is reproducible
    assert result_1['LR'].equals(result_2['LR'])
    
    df = result_1['LR']
    assert 'Feature' in df.columns
    assert 'Mean Global SHAP' in df.columns
    assert 'Standard deviation of Global SHAP' in df.columns
    assert 'Coefficient of Variation' in df.columns
    assert len(df) == 3 

def test_feature_robustness_assessment():

    intra_model_assessment_results = {
        'LR': pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Mean Global SHAP': [7.0, 4.0, 2.0],
            'Standard deviation of Global SHAP': [0.565886, 0.333058, 0.119945],
            'Coefficient of Variation': [0.026336, 0.021142, 0.922838]
            }),
        'RF': pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Mean Global SHAP': [1.0, 2.0, 3.0],
            'Standard deviation of Global SHAP': [0.1, 0.2, 0.3],
            'Coefficient of Variation': [0.02, 0.03, 0.24]
            })
        }

    global_shap_rankings = {
        'LR': pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Mean_Abs_SHAP': [3.0, 1.0, 2.0],
            'SHAP_Ranking': [1, 3, 2]
            }),
        'RF': pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'Mean_Abs_SHAP': [4.0, 3.0, 2.0],
            'SHAP_Ranking': [1, 2, 3]
            })
        }
    result = assess_features_robustness(global_shap_rankings, intra_model_assessment_results, 2)

    row_a = result[result['Feature'] == 'A'].iloc[0]

    assert row_a['Retained'] == 'Yes'
    assert row_a['CV stable'] == 'Yes'
    assert row_a['In Top-2 intersection'] == 'Yes'
    assert row_a['Mean Absolute SHAP'] == 3.5

    row_b = result[result['Feature'] == 'B'].iloc[0]

    assert row_b['Retained'] == 'No'
    assert row_b['CV stable'] == 'Yes'
    assert row_b['In Top-2 intersection'] == 'No'
    assert row_b['Mean Absolute SHAP'] == 2.0 

    row_c = result[result['Feature'] == 'C'].iloc[0]

    assert row_c['Retained'] == 'No'
    assert row_c['CV stable'] == 'No'
    assert row_c['In Top-2 intersection'] == 'No'
    assert row_c['Mean Absolute SHAP'] == 2.0
