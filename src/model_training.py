import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.feature_selection import select_top_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

### === REGRESSION ===

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
    y_predicted = pipeline.predict(X_test)

    n = X_test.shape[0]
    p = X_test.shape[1]
    
    #Results
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_test, y_predicted)          
    rmse = np.sqrt(mse)                               
    mae = mean_absolute_error(y_test, y_predicted)

    coefs = pipeline.named_steps['model'].best_estimator_.coef_
    number_of_features_selected = sum(abs(coefs) > 1e-6) #  Excludes features shrunk to zero (coefficients < 1e-6 carry zero predictive power). REFERENCE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 

    print("=" * 50)
    print("LINEAR REGRESSION - RESULTS")
    print("=" * 50)
    print(f"R²   : {r2:.4f}")
    print(f"R²_adj: {r2_adj:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    best_alpha = pipeline.named_steps['model'].best_params_['alpha']
    print(f"best alpha: {best_alpha}")
    print()
    print(f"Features selected: {number_of_features_selected}/{p}")


### === CLASSIFICATION ===
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
    y_predicted = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)

    print("=" * 50)
    print("LOGISTIC REGRESSION - RESULTS")
    print("=" * 50)
    print(f"accuracy: {accuracy:.4f}")
    print()
    print(f"classification report:")
    print(classification_report(y_test, y_predicted))
    print()
    print(f"confusion matrix:")
    print(confusion_matrix(y_test, y_predicted))
    print()
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")



def run_models(task, X_train, y_train, X_test, y_test, final_features_ranking, feature_names):

    for i in [5, 10, 15, 20]:

        top_k_features, _ = select_top_features(final_features_ranking, i)
        feature_to_index = {name: i for i, name in enumerate(feature_names)}
        indices = [feature_to_index[f] for f in top_k_features]

        X_train_with_top_features = X_train[:, indices]
        X_test_with_top_features = X_test[:, indices]

        if task == 'regression':
            print(f"Linear regression Model for {i} features")
            linear_regression(X_train_with_top_features, y_train, X_test_with_top_features, y_test)

        if task == 'classification':
            print(f"Logistic regression Model for {i} features")
            logistic_regression(X_train_with_top_features, y_train, X_test_with_top_features, y_test)
