import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from src.feature_selection import select_top_features
from sklearn.utils.class_weight import compute_sample_weight


def run_models(X_train, y_train, X_test, y_test, model_names, task):
    """
    Run multiple models for regression or classification and return a dictionary with trained models, predictions and all metrics.
    """
    
    results = {}
    
    if task == "regression":
        
        for name in model_names:
            print(f"\n{'='*60}")
            print(f"Training {name} (Regressor)")
            print('='*60)
            
            if name == "LR":
                alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                grid = GridSearchCV(
                    Lasso(random_state=7, max_iter=5000),
                    {'alpha': alphas},
                    cv=5,
                    scoring='r2'
                )
                model = Pipeline([("scaler", StandardScaler()), ("model", grid)])
                is_lasso = True
                
            elif name == "RF":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html 
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__n_estimators': [100],
                    'model__max_depth': [8],
                    'model__min_samples_split': [10, 20],
                    'model__min_samples_leaf': [12],
                    'model__max_features': ['sqrt'],
                    'model__ccp_alpha': [0.02, 0.05] #required to reduce strong overfitting observed
                }

                # No scaler needed for RF
                pipeline = Pipeline([("model", RandomForestRegressor(random_state=7, n_jobs=-1))])
                model = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1) # cv = 3 due to limited computational power
                is_lasso = False
                
            elif name == "XGBoost":
                # REFERENCE: https://xgboost.readthedocs.io/en/stable/parameter.html 
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__n_estimators': [50, 80],           
                    'model__max_depth': [3, 4],                
                    'model__learning_rate': [0.01, 0.03],      
                    'model__subsample': [0.5, 0.6],
                    'model__colsample_bytree': [0.4, 0.6],
                    'model__reg_alpha': [1.0, 2.0],            
                    'model__reg_lambda': [1.0, 2.0]
                }

                pipeline = Pipeline([("model", XGBRegressor(random_state=7, n_jobs=-1))])
                model = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1)
                is_lasso = False
                
            elif name == "SVR":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__kernel': ['linear', 'rbf'],
                    'model__C': [0.1, 1, 10, 100],
                    'model__gamma': ['scale', 0.01, 0.1],
                    'model__epsilon': [0.01, 0.1, 0.2]
                }

                pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVR())])
                model = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1)
                is_lasso = False
                
            else:
                print(f"Unknown model: {name}. Skipping.")
                continue
            
            # Train
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            n = X_test.shape[0]
            p = X_test.shape[1]
            
            train_r2 = r2_score(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            test_r2 = r2_score(y_test, y_test_pred)
            test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1) # # REFERENCE: https://www.datacamp.com/tutorial/adjusted-r-squared  
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # For Lasso only
            if is_lasso:
                # Get coefficients from the pipeline
                coefs = model.named_steps['model'].best_estimator_.coef_
                n_selected = sum(abs(coefs) > 1e-6) #  Excludes features shrunk to zero (coefficients < 1e-6 carry zero predictive power). REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 
                best_alpha = model.named_steps['model'].best_params_['alpha']
                print(f"Best alpha        : {best_alpha}")
                print(f"Features selected : {n_selected}/{p}")
            
            # Display results
            print("=" * 50)
            print(f"{name} (REGRESSION) - RESULTS")
            print("=" * 50)
            print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
            print("-" * 50)
            print(f"{'R²':<12} {train_r2:>10.4f} {test_r2:>10.4f} {train_r2 - test_r2:>10.4f}")
            print(f"{'MSE':<12} {train_mse:>10.2f} {test_mse:>10.2f} {train_mse - test_mse:>10.2f}")
            print(f"{'RMSE':<12} {train_rmse:>10.4f} {test_rmse:>10.4f} {train_rmse - test_rmse:>10.4f}")
            print(f"{'MAE':<12} {train_mae:>10.4f} {test_mae:>10.4f} {train_mae - test_mae:>10.4f}")
            print("-" * 50)
            print(f"R²_adj (test)     : {test_r2_adj:.4f}")
            
            status, detail, _, _ = check_overfitting_regression(train_r2, test_r2, test_r2_adj)
            print(f"\n{status}: {detail}")
            
            # Store all metrics in results dictionary for later aggregation
            results[name] = {
                'model': model,
                'y_pred': y_test_pred,
                # Training metrics
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                # Test metrics
                'test_r2': test_r2,
                'test_r2_adj': test_r2_adj,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                # Gap
                'gap_r2': train_r2 - test_r2,
                'gap': train_r2 - test_r2,
                # Lasso-specific
                'n_selected': n_selected if is_lasso else None,
                'best_alpha': best_alpha if is_lasso else None
            }
    
    elif task == "classification":
        
        # Store original y for later use (before encoding)
        y_train_original = y_train.copy() if hasattr(y_train, 'copy') else y_train
        y_test_original = y_test.copy() if hasattr(y_test, 'copy') else y_test
        
        for name in model_names:
            print(f"\n{'='*60}")
            print(f"Training {name} (Classifier)")
            print('='*60)
            
            # Reset y for each model (except XGBoost which needs encoding)
            y_train_use = y_train_original
            y_test_use = y_test_original
            encoder = None
            
            if name == "LR":
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(
                        max_iter=2000,
                        random_state=7,
                    ))
                ])
                parameters = {
                    'classifier__C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    'classifier__class_weight': ['balanced'],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga', 'lbfgs']
                }
                model = GridSearchCV(
                    pipeline,
                    parameters,
                    cv=5,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                
            elif name == "RF":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__n_estimators': [100],
                    'model__max_depth': [6, 8],              
                    'model__min_samples_split': [20, 50, 100],   
                    'model__min_samples_leaf': [20, 50],     
                    'model__max_features': ['sqrt'],
                    'model__class_weight': ['balanced'],
                    'model__ccp_alpha': [0.005, 0.01, 0.02]
                }

                pipeline = Pipeline([("model", RandomForestClassifier(random_state=7, n_jobs=-1))])

                model = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter=12,
                    cv=3,
                    scoring='f1_weighted',  # Important for imbalanced classes
                    n_jobs=-1,
                    verbose=1,
                    random_state=7
                )
                
            elif name == "XGBoost":
                # XGBoost requires numeric labels for classification. Thus, string labels (e.g., 'High Achievers', 'Low Proficient') must be encoded as integers.
                encoder = LabelEncoder()
                y_train_use = encoder.fit_transform(y_train_original)
                y_test_use = encoder.transform(y_test_original)
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_use)
                
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                # REFERENCE: https://www.datacamp.com/tutorial/ensemble-learning-python-guide?dc_referrer=https%3A%2F%2Fwww.google.com%2F 
                parameters = {
                    'model__n_estimators': [100, 200],                  
                    'model__max_depth': [3, 4],                    
                    'model__learning_rate': [0.01, 0.03, 0.05],          
                    'model__subsample': [0.6, 0.8],                     
                    'model__colsample_bytree': [0.6, 0.8],              
                    'model__gamma': [5, 10, 20],                        
                    'model__reg_alpha': [5.0, 10.0],                
                    'model__reg_lambda': [5.0, 10.0],               
                    'model__min_child_weight': [5, 10]              
                }

                pipeline = Pipeline([
                    ("model", XGBClassifier(random_state=7, n_jobs=-1, eval_metric='mlogloss'))
                ])

                model = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter=15,
                    cv=3,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1,
                    random_state=7
                )
                
            elif name == "SVC":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC 
                # Those final hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                        'model__kernel': ['rbf', 'poly'],          
                        'model__C': [0.1, 1, 10],                  
                        'model__gamma': ['scale', 'auto', 0.1],
                        'model__degree': [2],                   
                        'model__class_weight': ['balanced']
                    }

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),  # SVC is sensitive to feature scaling
                    ("model", SVC(random_state=7, probability=True))
                ])

                model = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter = 8,
                    cv=3,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1,
                    random_state=7
                )
                
            else:
                print(f"Unknown model: {name}. Skipping.")
                continue
            
            # Train

            if name == "XGBoost" and encoder is not None:
                model.fit(X_train, y_train_use, model__sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train_use)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # If XGBoost, decode predictions for display
            if name == "XGBoost" and encoder is not None:
                y_test_pred_display = encoder.inverse_transform(y_test_pred)
                y_test_true_display = encoder.inverse_transform(y_test_use)
            else:
                y_test_pred_display = y_test_pred
                y_test_true_display = y_test_use
            
            # Metrics
            train_accuracy = accuracy_score(y_train_use, y_train_pred)
            test_accuracy = accuracy_score(y_test_use, y_test_pred)
            train_f1_weighted = f1_score(y_train_use, y_train_pred, average='weighted')
            test_f1_weighted = f1_score(y_test_use, y_test_pred, average='weighted')
            train_f1_macro = f1_score(y_train_use, y_train_pred, average='macro')
            test_f1_macro = f1_score(y_test_use, y_test_pred, average='macro')
            
            # Precision and Recall (macro)
            test_precision_weighted = precision_score(y_test_use, y_test_pred, average='weighted')
            test_precision_macro = precision_score(y_test_use, y_test_pred, average='macro')
            test_recall_weighted = recall_score(y_test_use, y_test_pred, average='weighted')
            test_recall_macro = recall_score(y_test_use, y_test_pred, average='macro')
            
            print("=" * 50)
            print(f"{name} (CLASSIFICATION) - RESULTS")
            print("=" * 50)
            print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
            print("-" * 50)
            print(f"{'Accuracy':<15} {train_accuracy:>10.4f} {test_accuracy:>10.4f} {train_accuracy - test_accuracy:>10.4f}")
            print(f"{'F1 (weighted)':<15} {train_f1_weighted:>10.4f} {test_f1_weighted:>10.4f} {train_f1_weighted - test_f1_weighted:>10.4f}")
            print("-" * 50)
            
            # Classification report with original labels
            print("\nClassification Report (Test):")
            print(classification_report(y_test_true_display, y_test_pred_display))
            print("\nConfusion Matrix (Test):")
            print(confusion_matrix(y_test_true_display, y_test_pred_display))
            print()
            print(f"Best parameters: {model.best_params_}")
            print(f"Best CV F1-score: {model.best_score_:.4f}")
            
            status, detail, _, _ = check_overfitting_classification(train_accuracy, test_accuracy, train_f1_weighted, test_f1_weighted)
            print(f"\n{status}: {detail}")
            
            results[name] = {
                'model': model,
                'y_pred': y_test_pred_display,
                'train_accuracy': train_accuracy,
                'train_f1_weighted': train_f1_weighted,
                'train_f1_macro': train_f1_macro,
                'test_accuracy': test_accuracy,
                'test_f1_weighted': test_f1_weighted,
                'test_f1_macro': test_f1_macro,
                'test_precision_weighted': test_precision_weighted,
                'test_precision_macro': test_precision_macro,
                'test_recall_weighted': test_recall_weighted,
                'test_recall_macro': test_recall_macro,
                'gap_accuracy': train_accuracy - test_accuracy,
                'gap_f1': train_f1_weighted - test_f1_weighted,
                'gap': max(train_accuracy - test_accuracy, train_f1_weighted - test_f1_weighted)
            }
    
    else:
        print("Invalid task. Choose 'regression' or 'classification'.")
    
    return results

def check_overfitting_regression(train_r2, test_r2, test_r2_adj=None):

    # Overfitting check based on heuristics (INSPIRATION: https://datascience.stackexchange.com/questions/77298/how-many-ways-are-there-to-check-model-overfitting)
    
    gap = train_r2 - test_r2
    
    if test_r2_adj is not None:
        adj_gap = train_r2 - test_r2_adj
    else:
        adj_gap = gap

    # # A 10% R² gap is a clear overfitting signal, 5% can be considered as a warning zone whiles negative gaps below -5% are rare enough to be noted as positive.
    if gap > 0.10:
        status = "Overfitting"
        detail = f"R² gap = {gap:.4f} (> 0.10)"
    elif gap > 0.05:
        status = "Mild overfitting"
        detail = f"R² gap = {gap:.4f}"
    elif gap < -0.05:
        status = "Good generalisation (Test R² > Train R²)"
        detail = f"R² gap = {gap:.4f}"
    else:
        status = "Good generalisation"
        detail = f"R² gap = {gap:.4f}"
    
    return status, detail, gap, adj_gap

def check_overfitting_classification(train_acc, test_acc, train_f1, test_f1):

    # Overfitting check based on empirical classification heuristics.
    # In that situation, the threshold is stricter (0.05) than regression (0.10) because classification 
    # metrics are strictly bounded between 0 and 1. This means that in such situation, a 5% drop 
    # represents a critical loss of operational predictive power.
    acc_gap = train_acc - test_acc
    f1_gap = train_f1 - test_f1
    
    gap = max(acc_gap, f1_gap)
    
    if gap > 0.05:
        status = "Overfitting"
        detail = f"Acc gap = {acc_gap:.4f}, F1 gap = {f1_gap:.4f}"
    elif gap > 0.02:
        status = "Mild overfitting"
        detail = f"Acc gap = {acc_gap:.4f}, F1 gap = {f1_gap:.4f}"
    elif gap < -0.03: # A negative gap exceeding 3% is relatively rare in practice and suggests that the model generalises surprisingly well.
        status = "Good generalisation (Test > Train)"
        detail = f"Gap = {gap:.4f}"
    else:
        status = "Good generalisation"
        detail = f"Gap = {gap:.4f}"
    
    return status, detail, acc_gap, f1_gap

def evaluate_k_values(X_train, y_train, X_test, y_test, final_features_ranking, feature_names, model_names, task, k_values=[5, 10, 15, 20]):
    """
    Evaluate models for different values of k (number of features) and return a DataFrame + a dictionary with all results.
    
    Returns:
    - results_df: DataFrame with all metrics for each model and k
    - all_results: Complete dictionary with models, predictions, and features
    """
    
    all_results = {}
    rows = []
    
    print("\n" + "=" * 80)
    print(f"EVALUATING MODELS FOR DIFFERENT K VALUES ({task.upper()})")
    print("=" * 80)
    
    for k in k_values:
        print(f"\n{'#'*60}")
        print(f"# TESTING K = {k} FEATURES")
        print(f"{'#'*60}")
        
        # Select top k features
        top_k_features, _ = select_top_features(final_features_ranking, k)
        feature_to_index = {name: i for i, name in enumerate(feature_names)}
        indices = [feature_to_index[f] for f in top_k_features]
        
        X_train_k = X_train[:, indices]
        X_test_k = X_test[:, indices]
        
        # Display selected features
        print(f"\nTop {k} features selected:")
        for i, f in enumerate(top_k_features, 1):
            print(f"  {i}. {f}")
        
        # Run models
        results_k = run_models(
            X_train_k, y_train, X_test_k, y_test,
            model_names=model_names,
            task=task
        )
        
        all_results[k] = {
            'results': results_k,
            'features': top_k_features,
            'indices': indices
        }
        
        # Extract metrics for each model
        for name, result in results_k.items():
            row = {'k': k, 'Model': name}
            
            if task == 'regression':
                row['R²_train'] = result.get('train_r2')
                row['R²_test'] = result.get('test_r2')
                row['R²_adj'] = result.get('test_r2_adj')
                row['MSE_train'] = result.get('train_mse')
                row['MSE_test'] = result.get('test_mse')
                row['RMSE_train'] = result.get('train_rmse')
                row['RMSE_test'] = result.get('test_rmse')
                row['MAE_train'] = result.get('train_mae')
                row['MAE_test'] = result.get('test_mae')
                row['Gap_R²'] = result.get('gap_r2')
                
                if name == 'LR':
                    row['Features_Selected'] = result.get('n_selected')
                    row['Best_Alpha'] = result.get('best_alpha')
                
            else:  # classification
                row['Accuracy_train'] = result.get('train_accuracy')
                row['Accuracy_test'] = result.get('test_accuracy')
                row['F1_weighted_train'] = result.get('train_f1_weighted')
                row['F1_weighted_test'] = result.get('test_f1_weighted')
                row['F1_macro_train'] = result.get('train_f1_macro')
                row['F1_macro_test'] = result.get('test_f1_macro')
                row['Precision_weighted_test'] = result.get('test_precision_weighted')
                row['Precision_macro_test'] = result.get('test_precision_macro')
                row['Recall_weighted_test'] = result.get('test_recall_weighted')
                row['Recall_macro_test'] = result.get('test_recall_macro')
                row['Gap_Accuracy'] = result.get('gap_accuracy')
                row['Gap_F1'] = result.get('gap_f1')
            
            rows.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(rows)
    
    # Display summary
    print("\n" + "=" * 80)
    print(f"SUMMARY - {task.upper()} METRICS")
    print("=" * 80)
    
    if task == 'regression':
        display_cols = ['k', 'Model', 'R²_test', 'RMSE_test', 'MAE_test', 'Gap_R²']
        if 'Features_Selected' in results_df.columns:
            display_cols.append('Features_Selected')
        print(results_df[display_cols].to_string(index=False))
    else:
        print(results_df[['k', 'Model', 'Accuracy_test', 'F1_weighted_test', 'Gap_Accuracy']].to_string(index=False))
    
    return results_df, all_results