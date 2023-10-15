import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

def generate_model(df):
    # Separate features (X) and target (y)
    X = df.drop(labels=['target_column'], axis=1)  # Replace 'target_column' with your target column name
    y = df['target_column']  # Replace 'target_column' with your target column name

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an XGBoost model
    model = xgb.XGBClassifier()

    # Create a RobustScaler for feature scaling
    scaler = RobustScaler()

    # Create a pipeline with feature scaling and the XGBoost model
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])

    # Define hyperparameter grid for tuning (adjust as needed)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.1, 0.01, 0.001],
        # Add more hyperparameters as needed
    }

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=kf)

    # Fit the model and perform hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return best_model, scaler, best_params, accuracy


best_model, scaler, best_params, accuracy = generate_model(df)
print("Best Model:", best_model)
print("Best Scaler:", scaler)
print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)
