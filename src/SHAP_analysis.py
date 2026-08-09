import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr

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

def local_and_global_shap_values_calculator(rashomon_set_dict, X_train, X_test, feature_names, task='regression'):
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
                X_train_for_shap = scaler.transform(X_train) if scaler else X_train
                X_test_for_shap = scaler.transform(X_test) if scaler else X_test
                explainer = shap.LinearExplainer(best_estimator.named_steps['model'], X_train_for_shap, feature_perturbation="correlation_dependent") # helps keeping realism in profile computation by computing "smart" conditional expectations. SOURCE: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Math%20behind%20LinearExplainer%20with%20correlation%20feature%20perturbation.html 
                shap_values = explainer.shap_values(X_test_for_shap)

            elif model_name == "RF" or model_name == "XGBoost":
                explainer = shap.TreeExplainer(best_estimator.named_steps['model'])
                shap_values = explainer.shap_values(X_test)

            elif model_name == "SVR":

                #Background_sample and X_test_sample processes are required to limit the computational cost of this operation.
                # shap.kmeans() is preferred over shap.sample() as it selects better representative data points.
                max_samples = 30 

                if X_test.shape[0] > max_samples:

                    if hasattr(X_test, "sample"):
                        X_test_sample = X_test.sample(n=max_samples, random_state=7)

                    else:
                        indices = np.random.choice(X_test.shape[0], size=max_samples, replace=False)
                        X_test_sample = X_test[indices]
                else:
                    X_test_sample = X_test
                    
                scaler = best_estimator.named_steps.get('scaler', None)
                X_train_for_shap= scaler.transform(X_train) if scaler else X_train
                X_test_for_shap = scaler.transform(X_test_sample) if scaler else X_test_sample

                background_sample = shap.kmeans(X_train_for_shap, 50)
                explainer = shap.KernelExplainer(best_estimator.named_steps['model'].predict, background_sample)
                shap_values = explainer.shap_values(X_test_for_shap)

            else:
                raise ValueError(f"Unknown model: {model_name}. The global_shap_values_calculator is calibrated to compute only SHAP values for regression models.")

            # SOURCE: https://stackoverflow.com/questions/77474923/calculate-the-mean-of-absolute-shap-values-across-all-classes 
            feature_importance_overall = np.mean(np.abs(shap_values), axis=0)

            ranking_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean_Abs_SHAP': feature_importance_overall
                }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        
            ranking_df['SHAP_Ranking'] = ranking_df.index + 1

        global_mean_absolute_shap_rankings[model_name] = ranking_df
        individual_shap_values_for_every_model[model_name] = {'shap_values': shap_values,'X_test_subset': X_test_for_shap if model_name in ["LR", "SVR"] else X_test}
    
    return global_mean_absolute_shap_rankings, individual_shap_values_for_every_model


def inter_model_concordance_assessment(shap_global_rankings_dic, top_k_features_concordance):

    results=[]
    model_names = list(shap_global_rankings_dic.keys())

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):

            df_model_i = shap_global_rankings_dic[model_names[i]]
            df_model_j = shap_global_rankings_dic[model_names[j]]

            df_merged = pd.merge(df_model_i[['Feature','SHAP_Ranking']], df_model_j[['Feature','SHAP_Ranking']], on='Feature', suffixes=(f'_{model_names[i]}', f'_{model_names[j]}'))
            kendall_coeff, kendall_p_value = kendalltau(df_merged[f'SHAP_Ranking_{model_names[i]}'], df_merged[f'SHAP_Ranking_{model_names[j]}'])
            spearman_coeff, spearman_p_value = spearmanr(df_merged[f'SHAP_Ranking_{model_names[i]}'], df_merged[f'SHAP_Ranking_{model_names[j]}'])

            top_k_features_model_i = set(df_model_i.head(top_k_features_concordance)['Feature'])
            top_k_features_model_j = set(df_model_j.head(top_k_features_concordance)['Feature'])
            overlap_ratio = len(top_k_features_model_i & top_k_features_model_j) / top_k_features_concordance

            results.append({
                'Model 1': model_names[i],
                'Model 2': model_names[j],
                'Kendall_coeff': kendall_coeff,
                'Kendall p value': kendall_p_value,
                'Spearman coeff': spearman_coeff,
                'Spearman p value': spearman_p_value,
                'Overlap ratio': overlap_ratio
            })

    return pd.DataFrame(results)



                       

    
        
