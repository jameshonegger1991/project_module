import numpy as np

from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from src.feature_selection import select_top_features

### === REGRESSION ===

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

def linear_regression(X_train, y_train, X_test, y_test):

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    grid = GridSearchCV(
        Lasso(random_state=7, max_iter=5000),
        {'alpha': alphas},
        cv=5,
        scoring='r2'
    )

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", grid)])
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    n = X_test.shape[0]
    p = X_test.shape[1]

    # Metrics for train split
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    # Metrics for test split
    test_r2 = r2_score(y_test, y_test_pred)
    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1) # # REFERENCE: https://www.datacamp.com/tutorial/adjusted-r-squared  
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    coefs = pipeline.named_steps['model'].best_estimator_.coef_
    number_of_features_selected = sum(abs(coefs) > 1e-6) #  Excludes features shrunk to zero (coefficients < 1e-6 carry zero predictive power). REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 

    print("=" * 50)
    print("LASSO REGRESSION - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'R²':<12} {train_r2:>10.4f} {test_r2:>10.4f} {train_r2 - test_r2:>10.4f}")
    print(f"{'MSE':<12} {train_mse:>10.2f} {test_mse:>10.2f} {train_mse - test_mse:>10.2f}")
    print(f"{'RMSE':<12} {train_rmse:>10.4f} {test_rmse:>10.4f} {train_rmse - test_rmse:>10.4f}")
    print(f"{'MAE':<12} {train_mae:>10.4f} {test_mae:>10.4f} {train_mae - test_mae:>10.4f}")
    print("-" * 50)
    print(f"R²_adj (test)     : {test_r2_adj:.4f}")
    
    best_alpha = pipeline.named_steps['model'].best_params_['alpha']
    print(f"Best alpha        : {best_alpha}")
    print(f"Features selected : {number_of_features_selected}/{p}")

    status, detail, _, _ = check_overfitting_regression(train_r2, test_r2, test_r2_adj)
    print(f"\n{status}: {detail}")

def random_forest_regression(X_train, y_train, X_test, y_test):

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
    pipeline = Pipeline([("model", RandomForestRegressor( random_state=7, n_jobs=-1))])
    grid_search = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1) # cv = 3 due to limited computational power
    grid_search.fit(X_train, y_train)

    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)
    # best_model = grid_search.best_estimator_

    n = X_test.shape[0]
    p = X_test.shape[1]
    
    # Train metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "=" * 50)
    print("RANDOM FOREST REGRESSION - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'R²':<12} {train_r2:>10.4f} {test_r2:>10.4f} {train_r2 - test_r2:>10.4f}")
    print(f"{'MSE':<12} {train_mse:>10.2f} {test_mse:>10.2f} {train_mse - test_mse:>10.2f}")
    print(f"{'RMSE':<12} {train_rmse:>10.4f} {test_rmse:>10.4f} {train_rmse - test_rmse:>10.4f}")
    print(f"{'MAE':<12} {train_mae:>10.4f} {test_mae:>10.4f} {train_mae - test_mae:>10.4f}")
    print("-" * 50)
    print(f"R²_adj (test)     : {test_r2_adj:.4f}")

    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV R²: {grid_search.best_score_:.4f}")
    
    # Overfitting detection
    status, detail, _, _ = check_overfitting_regression(train_r2, test_r2, test_r2_adj)
    print(f"\n{status}: {detail}")

def XGBoost_regressor(X_train, y_train, X_test, y_test):

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
    grid_search = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)
    # best_model = grid_search.best_estimator_

    n = X_test.shape[0]
    p = X_test.shape[1]
    
    # Train metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "=" * 50)
    print("XGBOOST REGRESSION - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'R²':<12} {train_r2:>10.4f} {test_r2:>10.4f} {train_r2 - test_r2:>10.4f}")
    print(f"{'MSE':<12} {train_mse:>10.2f} {test_mse:>10.2f} {train_mse - test_mse:>10.2f}")
    print(f"{'RMSE':<12} {train_rmse:>10.4f} {test_rmse:>10.4f} {train_rmse - test_rmse:>10.4f}")
    print(f"{'MAE':<12} {train_mae:>10.4f} {test_mae:>10.4f} {train_mae - test_mae:>10.4f}")
    print("-" * 50)
    print(f"R²_adj (test)     : {test_r2_adj:.4f}")

    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV R²: {grid_search.best_score_:.4f}")
    
    # Overfitting detection
    status, detail, _, _ = check_overfitting_regression(train_r2, test_r2, test_r2_adj)
    print(f"\n{status}: {detail}")

def SVR_regressor(X_train, y_train, X_test, y_test):

    # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    # The hyperparameters were selected after iterative trial-and-error experimentation.
    parameters = {
        'model__kernel': ['linear', 'rbf'],
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 0.01, 0.1],
        'model__epsilon': [0.01, 0.1, 0.2]
    }

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVR())])
    grid_search = GridSearchCV(pipeline, parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)
    # best_model = grid_search.best_estimator_

    n = X_test.shape[0]
    p = X_test.shape[1]
    
    # Train metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "=" * 50)
    print("SVR - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'R²':<12} {train_r2:>10.4f} {test_r2:>10.4f} {train_r2 - test_r2:>10.4f}")
    print(f"{'MSE':<12} {train_mse:>10.2f} {test_mse:>10.2f} {train_mse - test_mse:>10.2f}")
    print(f"{'RMSE':<12} {train_rmse:>10.4f} {test_rmse:>10.4f} {train_rmse - test_rmse:>10.4f}")
    print(f"{'MAE':<12} {train_mae:>10.4f} {test_mae:>10.4f} {train_mae - test_mae:>10.4f}")
    print("-" * 50)
    print(f"R²_adj (test)     : {test_r2_adj:.4f}")

    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV R²: {grid_search.best_score_:.4f}")
    
    # Overfitting detection
    status, detail, _, _ = check_overfitting_regression(train_r2, test_r2, test_r2_adj)
    print(f"\n{status}: {detail}")


### === CLASSIFICATION ===

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

def logistic_regression(X_train, y_train, X_test, y_test):

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

    grid_search = GridSearchCV(
            pipeline,
            parameters,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
    
    grid_search.fit(X_train, y_train)

    # Predictions on train and test
    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)
    
    # Train metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print("=" * 50)
    print("LOGISTIC REGRESSION - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {train_accuracy:>10.4f} {test_accuracy:>10.4f} {train_accuracy - test_accuracy:>10.4f}")
    print(f"{'F1 (weighted)':<15} {train_f1:>10.4f} {test_f1:>10.4f} {train_f1 - test_f1:>10.4f}")
    print("-" * 50)
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    print()
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
    
    status, detail, _, _ = check_overfitting_classification(train_accuracy, test_accuracy, train_f1, test_f1)
    print(f"\n{status}: {detail}")

def random_forest_classifier(X_train, y_train, X_test, y_test):

    # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
    # The hyperparameters were selected after iterative trial-and-error experimentation.
    parameters = {
        'model__n_estimators': [100],
        'model__max_depth': [8, 12],
        'model__min_samples_split': [10, 20],
        'model__min_samples_leaf': [8, 12],
        'model__max_features': ['sqrt'],
        'model__class_weight': ['balanced'],
        'model__ccp_alpha': [0.005, 0.01]
    }

    pipeline = Pipeline([("model", RandomForestClassifier(random_state=7, n_jobs=-1))])

    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=3,
        scoring='f1_weighted',  # Important for imbalanced classes
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Predictions
    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print("=" * 50)
    print("RANDOM FOREST CLASSIFIER - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {train_accuracy:>10.4f} {test_accuracy:>10.4f} {train_accuracy - test_accuracy:>10.4f}")
    print(f"{'F1 (weighted)':<15} {train_f1:>10.4f} {test_f1:>10.4f} {train_f1 - test_f1:>10.4f}")
    print("-" * 50)

    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV F1: {grid_search.best_score_:.4f}")

    print()
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))

    # Overfitting detection
    status, detail, _, _ = check_overfitting_classification(train_accuracy, test_accuracy, train_f1, test_f1)
    print(f"\n{status}: {detail}")

def XGBoost_classifier(X_train, y_train, X_test, y_test):

    # XGBoost requires numeric labels for classification. Thus, string labels (e.g., 'High Achievers', 'Low Proficient') must be encoded as integers.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    # The hyperparameters were selected after iterative trial-and-error experimentation.
    # REFERENCE: https://www.datacamp.com/tutorial/ensemble-learning-python-guide?dc_referrer=https%3A%2F%2Fwww.google.com%2F 
    parameters = {
        'model__n_estimators': [30, 50],             
        'model__max_depth': [2, 3],                  
        'model__learning_rate': [0.01, 0.02],        
        'model__subsample': [0.4, 0.5],              
        'model__colsample_bytree': [0.4, 0.5],       
        'model__reg_alpha': [2.0, 5.0],              
        'model__reg_lambda': [2.0, 5.0]              
    }

    pipeline = Pipeline([
        ("model", XGBClassifier(random_state=7, n_jobs=-1, eval_metric='mlogloss'))
    ])

    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_encoded)

    y_train_pred_encoded = grid_search.predict(X_train)
    y_test_pred_encoded = grid_search.predict(X_test)

    # Metrics
    train_accuracy = accuracy_score(y_train_encoded, y_train_pred_encoded)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred_encoded)
    train_f1 = f1_score(y_train_encoded, y_train_pred_encoded, average='weighted')
    test_f1 = f1_score(y_test_encoded, y_test_pred_encoded, average='weighted')

    print("=" * 50)
    print("XGBoost CLASSIFIER - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {train_accuracy:>10.4f} {test_accuracy:>10.4f} {train_accuracy - test_accuracy:>10.4f}")
    print(f"{'F1 (weighted)':<15} {train_f1:>10.4f} {test_f1:>10.4f} {train_f1 - test_f1:>10.4f}")
    print("-" * 50)

    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV F1: {grid_search.best_score_:.4f}")

    print()
    print("\nClassification Report (Test):")
    print(classification_report(y_test_encoded, y_test_pred_encoded))
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test_encoded, y_test_pred_encoded))

    # Overfitting detection
    status, detail, _, _ = check_overfitting_classification(train_accuracy, test_accuracy, train_f1, test_f1)
    print(f"\n{status}: {detail}")

def SVC_classifier(X_train, y_train, X_test, y_test):
    
    # SVC requires numeric labels for classification. Thus, string labels (e.g., 'High Achievers', 'Low Proficient') must be encoded as integers.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC 
    # Those final hyperparameters were selected after iterative trial-and-error experimentation.
    parameters = {
        'model__kernel': ['linear'],           
        'model__C': [0.01, 0.1, 1.0],          
        'model__class_weight': ['balanced']
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # SVC is sensitive to feature scaling
        ("model", SVC(random_state=7, probability=True))
    ])

    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_encoded)

    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)

    # Train/test metrics
    train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    train_f1 = f1_score(y_train_encoded, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')

    print("=" * 50)
    print("SVC CLASSIFIER - RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {train_accuracy:>10.4f} {test_accuracy:>10.4f} {train_accuracy - test_accuracy:>10.4f}")
    print(f"{'F1 (weighted)':<15} {train_f1:>10.4f} {test_f1:>10.4f} {train_f1 - test_f1:>10.4f}")
    print("-" * 50)

    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV F1: {grid_search.best_score_:.4f}")

    print()
    print("\nClassification Report (Test):")
    print(classification_report(y_test_encoded, y_test_pred))
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test_encoded, y_test_pred))

    # Overfitting detection
    status, detail, _, _ = check_overfitting_classification(train_accuracy, test_accuracy, train_f1, test_f1)
    print(f"\n{status}: {detail}")


def run_models(task, X_train, y_train, X_test, y_test, final_features_ranking, feature_names):

    for i in [5, 10, 15, 20]:

        top_k_features, _ = select_top_features(final_features_ranking, i)
        feature_to_index = {name: i for i, name in enumerate(feature_names)}
        indices = [feature_to_index[f] for f in top_k_features]

        X_train_with_top_features = X_train[:, indices]
        X_test_with_top_features = X_test[:, indices]

        if task == 'regression':
            print()
            print(f"Linear regression Model for {i} features")
            linear_regression(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
            print()
            print(f"Random forest regressor model for {i} features")
            random_forest_regression(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
            print()
            print(f"XGBOOST Regressor model for {i} features")
            XGBoost_regressor(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
            print()
            print(f"SVR model for {i} features")
            SVR_regressor(X_train_with_top_features, y_train, X_test_with_top_features, y_test)

        if task == 'classification':
            print()
            print(f"Logistic regression Model for {i} features")
            logistic_regression(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
            print()
            print(f"Random forest Classifier model for {i} features")
            random_forest_classifier(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
            print()
            print(f"XGBOOST Classifier for {i} features")
            XGBoost_classifier(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
            print()
            print(f"SVR model for {i} features")
            SVC_classifier(X_train_with_top_features, y_train, X_test_with_top_features, y_test)

