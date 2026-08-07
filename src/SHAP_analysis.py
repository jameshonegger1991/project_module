import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

def rashomon_set_builder(model_training_all_results_dic, k_nbr_of_features_chosen, task: str = 'regression', rashomon_threshold: float = 0.05):
    """
    Constructs a Rashomon set from the model training all_results dictionary for a given k.
    """

    if model_training_all_results_dic is None or not isinstance(model_training_all_results_dic, dict):
        raise ValueError("model_training_all_results_dic must be a valid dictionary.")

    if task not in ['regression', 'classification']:
        raise ValueError("Invalid task. Task must be 'regression' or 'classification'.")

    if rashomon_threshold < 0:
        raise ValueError("rashomon_threshold must be non-negative.")

    if k_nbr_of_features_chosen not in model_training_all_results_dic:
        raise ValueError(f"No results found for k = {k_nbr_of_features_chosen} in the dictionary.")

    results_k = model_training_all_results_dic[k_nbr_of_features_chosen]['results']
    
    model_scores = {}
    for model_name, result in results_k.items():
        score = result.get('best_cv_score')
        if score is not None:
            model_scores[model_name] = score

    if not model_scores:
        raise ValueError(f"No CV scores found for k = {k_nbr_of_features_chosen}.")

    best_score = max(model_scores.values())
    
    # Absolute threshold alignmed with the project proposal (Delta <= 0.05)
    lowest_score_acceptable = best_score - rashomon_threshold

    rashomon_set_dict = {}
    
    for model_name, score in model_scores.items():
        if score >= lowest_score_acceptable:
            rashomon_set_dict[model_name] = results_k[model_name]

    metric_label = "best R² CV score" if task == 'regression' else "best F1 macro CV score"
    
    print(f"\n{'='*60}")
    print(f"RASHOMON SET CONSTRUCTION (k={k_nbr_of_features_chosen}, Task={task})")
    print(f"{'='*60}")
    print(f"\nBest CV Score achieved : {best_score:.4f}")
    print(f"Acceptable threshold   : >= {lowest_score_acceptable:.4f} (Δ <= {rashomon_threshold})")
    print(f"\nRashomon set contains the following models:")
    print()
    for model_name, res_dict in rashomon_set_dict.items():
        score = res_dict.get('best_cv_score')
        print(f"  • {model_name:<10} : {metric_label} = {score:.4f}")
        
    print(f"\n{'='*60}\n")

    return rashomon_set_dict

def global_shap_values_calculator(rashomon_set_dict, X_train, X_test, feature_names, task='regression'):
# INSPIRATION: - https://stackoverflow.com/questions/72599807/how-to-extract-the-most-important-features-from-a-ml-model-using-shap-why-are 
#              - https://www.geeksforgeeks.org/machine-learning/shap-with-a-linear-svc-model-from-sklearn-using-pipeline/
    
    global_mean_absolute_shap_rankings = {}
    individual_shap_values_for_every_model = {}

    for model_name, results in rashomon_set_dict.items():

        model_grid_search = results['model']
        best_estimator = model_grid_search.best_estimator_

        if task == 'regression':
            if model_name == "LR":
                scaler = best_estimator.named_steps.get('scaler', None)
                X_train_for_shap= scaler.transform(X_train) if scaler else X_train
                X_test_for_shap = scaler.transform(X_test) if scaler else X_test
                explainer = shap.LinearExplainer(best_estimator.named_steps['model'], X_train_for_shap, feature_perturbation="correlation_dependent") # helps keeping realism in profile computation by computing "smart" conditional expectations. SOURCE: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Math%20behind%20LinearExplainer%20with%20correlation%20feature%20perturbation.html 
                shap_values = explainer.shap_values(X_test_for_shap)

            elif model_name == "RF":
                continue
            elif model_name == "XGBoost":
                continue
            elif model_name == "SVR":
                continue
            # SOURCE: https://stackoverflow.com/questions/77474923/calculate-the-mean-of-absolute-shap-values-across-all-classes 
            abs_shap_values = np.abs(shap_values)
            feature_importance_overall = np.mean(np.abs(shap_values), axis=0)

            ranking_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean_Abs_SHAP': feature_importance_overall
                }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        
            ranking_df['SHAP_Ranking'] = ranking_df.index + 1

        global_mean_absolute_shap_rankings[model_name] = ranking_df
        individual_shap_values_for_every_model[model_name] = shap_values
    
    return global_mean_absolute_shap_rankings, individual_shap_values_for_every_model