def rashomon_set_builder(model_training_all_results_dic, k_nbr_of_features_chosen, task: str = 'regression', rashomon_threshold: float = 0.05):
    """
    Constructs a Rashomon set from the model training all_results dictionary for a given k.
    Returns a dictionary where each model in the set maps to its full result dictionary (including the trained model).
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

