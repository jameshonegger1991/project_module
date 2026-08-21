import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from src.feature_selection import select_top_features
from sklearn.utils.class_weight import compute_sample_weight

from src.config import(
    RANDOM_STATE,
)

def run_models(X_train, y_train, X_test, y_test, model_names, task):
    """
    Trains and evaluates models with hyperparameter tuning (Grid/RandomizedSearch).
    Returns fitted models, predictions, metrics (train/test), best CV scores, and best parameters.

    Arguments:
    - X_train, y_train: Training features (after variance_threshold for the purpose of this project) and target. 
    - X_test, y_test: Test features (after variance_threshold for the purpose of this project) and target.
    - model_names: List of model names to run.
    - task: 'regression' or 'classification'.

    Returns:
    - dict: Results for each model.
    """
    
    results = {}
    
    if task == "regression":
        
        for name in model_names:
            print(f" > Training {name} (Regressor)...")
            
            if name == "LR":
                alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                
                pipeline = Pipeline([
                    ("scaler", StandardScaler()), 
                    ("model", Lasso(random_state=RANDOM_STATE, max_iter=5000))
                ])
                
                model = GridSearchCV(
                    pipeline,
                    {'model__alpha': alphas}, 
                    cv=5,
                    scoring='r2'
                )

                is_lasso = True
                
            elif name == "RF":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html 
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__n_estimators': [100],
                    'model__max_depth': [5, 6],
                    'model__min_samples_split': [10, 20],
                    'model__min_samples_leaf': [20, 30],
                    'model__max_features': ['sqrt'],
                    'model__ccp_alpha': [0.01, 0.02, 0.04] #required to reduce strong overfitting observed
                }

                # No scaler needed for RF
                pipeline = Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
                model = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=0) # cv = 3 due to limited computational power
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

                pipeline = Pipeline([("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
                model = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=0)
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
                model = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=0)
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
            
            if is_lasso:
                coefs = model.best_estimator_.named_steps['model'].coef_
                n_selected = sum(abs(coefs) > 1e-6)
                best_alpha = model.best_params_['model__alpha']
            
            results[name] = {
                'model': model,
                'y_pred': y_test_pred,
                'y_test_true': y_test, 
                'type': 'regression',

                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,

                'test_r2': test_r2,
                'test_r2_adj': test_r2_adj,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,

                'gap_r2': train_r2 - test_r2,
                'gap': train_r2 - test_r2,
                'best_cv_score': model.best_score_,

                'n_selected': n_selected if is_lasso else None,
                'best_alpha': best_alpha if is_lasso else None
            }
    
    elif task == "classification":
        
        y_train_original = y_train.copy() if hasattr(y_train, 'copy') else y_train
        y_test_original = y_test.copy() if hasattr(y_test, 'copy') else y_test
        
        for name in model_names:
            print(f" > Training {name} (Classifier)...")
            
            # Reset y for each model (except XGBoost which needs encoding)
            y_train_use = y_train_original
            y_test_use = y_test_original
            encoder = None
            
            if name == "LR":
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ))
                ])
                #Bug fixed with a list of dic for parameters (as lbfgs doesn't support l1)
                parameters = [
                    {
                        'classifier__C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                        'classifier__class_weight': ['balanced'],
                        'classifier__penalty': ['l2'],
                        'classifier__solver': ['lbfgs', 'liblinear', 'saga']
                    },
                    {
                        'classifier__C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                        'classifier__class_weight': ['balanced'],
                        'classifier__penalty': ['l1'],
                        'classifier__solver': ['liblinear', 'saga']
                    }
                ]
                model = GridSearchCV(
                    pipeline,
                    parameters,
                    cv=5,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=0
                )
                
            elif name == "RF":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__n_estimators': [200],
                    'model__max_depth': [6, 8],                 
                    'model__min_samples_split': [30, 50],        
                    'model__min_samples_leaf': [15, 25],        
                    'model__max_features': ['sqrt'],
                    'model__class_weight': ['balanced'],
                    'model__ccp_alpha': [0.001, 0.002]
                }

                pipeline = Pipeline([("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])

                model = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter=12,
                    cv=3,
                    scoring='f1_macro',  # Important for imbalanced classes
                    n_jobs=-1,
                    verbose=0,
                    random_state=RANDOM_STATE
                )
                
            elif name == "XGBoost":
                # XGBoost requires numeric labels for classification. 
                encoder = LabelEncoder()
                y_train_use = encoder.fit_transform(y_train_original)
                y_test_use = encoder.transform(y_test_original)
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_use)
                
                # The hyperparameters were selected after iterative trial-and-error experimentation.
                # REFERENCE: https://www.datacamp.com/tutorial/ensemble-learning-python-guide?dc_referrer=https%3A%2F%2Fwww.google.com%2F 
                parameters = {
                    'model__n_estimators': [100],                  
                    'model__max_depth': [3, 4],                  
                    'model__learning_rate': [0.03],          
                    'model__subsample': [0.8],                     
                    'model__colsample_bytree': [0.8],              
                    'model__gamma': [2, 5],                        
                    'model__reg_alpha': [5.0],                
                    'model__reg_lambda': [5.0, 10.0],            
                    'model__min_child_weight': [5, 10]             
                }

                pipeline = Pipeline([
                    ("model", XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'))
                ])

                model = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter=15,
                    cv=3,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=0,
                    random_state=RANDOM_STATE
                )
                
            elif name == "SVC":
                # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC 
                # Those final hyperparameters were selected after iterative trial-and-error experimentation.
                parameters = {
                    'model__kernel': ['rbf', 'poly'],          
                    'model__C': [0.1, 0.5, 1],                  
                    'model__gamma': ['scale', 'auto'],
                    'model__degree': [2],                   
                    'model__class_weight': ['balanced']
                }

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),  # SVC is sensitive to feature scaling
                    ("model", SVC(random_state=RANDOM_STATE, probability=True))
                ])

                model = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter = 8,
                    cv=3,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=0,
                    random_state=RANDOM_STATE
                )
                
            else:
                print(f"Unknown model: {name}. Skipping.")
                continue
            
            if name == "XGBoost" and encoder is not None:
                model.fit(X_train, y_train_use, model__sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train_use)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            if name == "XGBoost" and encoder is not None:
                y_test_pred_display = encoder.inverse_transform(y_test_pred)
                y_test_true_display = encoder.inverse_transform(y_test_use)
            else:
                y_test_pred_display = y_test_pred
                y_test_true_display = y_test_use
            
            train_accuracy = accuracy_score(y_train_use, y_train_pred)
            test_accuracy = accuracy_score(y_test_use, y_test_pred)
            train_f1_weighted = f1_score(y_train_use, y_train_pred, average='weighted')
            test_f1_weighted = f1_score(y_test_use, y_test_pred, average='weighted')
            train_f1_macro = f1_score(y_train_use, y_train_pred, average='macro')
            test_f1_macro = f1_score(y_test_use, y_test_pred, average='macro')
            
            # Precision and Recall (macro)
            test_precision_weighted = precision_score(y_test_use, y_test_pred, average='weighted')
            test_precision_macro = precision_score(y_test_use, y_test_pred, average='macro')
            train_precision_macro = precision_score(y_train_use, y_train_pred, average='macro')
            
            test_recall_weighted = recall_score(y_test_use, y_test_pred, average='weighted')
            test_recall_macro = recall_score(y_test_use, y_test_pred, average='macro')
            train_recall_macro = recall_score(y_train_use, y_train_pred, average='macro')
            
            results[name] = {
                'model': model,
                'y_pred': y_test_pred_display,
                'y_test_true': y_test_true_display,
                'type': 'classification',
                'train_accuracy': train_accuracy,
                'train_f1_weighted': train_f1_weighted,
                'train_f1_macro': train_f1_macro,
                'train_precision_macro': train_precision_macro,
                'train_recall_macro': train_recall_macro,
                'test_accuracy': test_accuracy,
                'test_f1_weighted': test_f1_weighted,
                'test_f1_macro': test_f1_macro,
                'test_precision_weighted': test_precision_weighted,
                'test_precision_macro': test_precision_macro,
                'test_recall_weighted': test_recall_weighted,
                'test_recall_macro': test_recall_macro,
                'gap_accuracy': train_accuracy - test_accuracy,
                'gap_f1': train_f1_macro - test_f1_macro,
                'gap': max(train_accuracy - test_accuracy, train_f1_weighted - test_f1_weighted),
                'best_params': model.best_params_,      
                'best_cv_score': model.best_score_
            }
    
    else:
        print("Invalid task. Choose 'regression' or 'classification'.")
    
    return results

def evaluate_k_values(X_train, y_train, X_test, y_test, final_features_ranking, feature_names, model_names, task, k_values=[5, 10, 15, 20]):
    """
    Evaluate models for different values of k (number of features) and return a DataFrame + a dictionary with all results.
    For each k, it selects the top k features and calls run_models() internally to get results.

    Arguments:
    - X_train, y_train, X_test, y_test: Train/test data.
    - final_features_ranking: Ranked list of features (used for ordering).
    - feature_names: All feature names matching the data columns.
    - model_names: List of models to evaluate.
    - task: 'regression' or 'classification'.
    - k_values: List of feature counts to test. Default [5, 10, 15, 20].

    Returns:
    -  tuple: Summary DataFrame of model performance across k values and a dictionary 
    containing the detailed results for each k.
    """
    all_results = {}
    rows = []
    
    for k in k_values:
        print(f"\n*** TRAINING MODELS WITH THE {k} TOP FEATURES ***")
        print()
        # Select top k features
        top_k_features, _ = select_top_features(final_features_ranking, k)
        feature_to_index = {name: i for i, name in enumerate(feature_names)}
        indices = [feature_to_index[f] for f in top_k_features]
        
        X_train_k = X_train[:, indices]
        X_test_k = X_test[:, indices]
        
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
            row['CV_Score'] = result.get('best_cv_score')
            
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
                row['Precision_macro_train'] = result.get('train_precision_macro')
                row['Recall_macro_train'] = result.get('train_recall_macro')
                row['Precision_weighted_test'] = result.get('test_precision_weighted')
                row['Precision_macro_test'] = result.get('test_precision_macro')
                row['Recall_weighted_test'] = result.get('test_recall_weighted')
                row['Recall_macro_test'] = result.get('test_recall_macro')
                row['Gap_Accuracy'] = result.get('gap_accuracy')
                row['Gap_F1'] = result.get('gap_f1')
            
            rows.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(rows)
    return results_df, all_results