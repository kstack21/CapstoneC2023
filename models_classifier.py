import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


def generate_model(df):
    """
    Train and evaluate multiple machine learning models using K-Fold cross-validation
    and hyperparameter tuning, and return the best model along with the scaler used for feature scaling.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the dataset with features and target variable.

    Returns:
    - best_model (sklearn.pipeline.Pipeline): The best-performing machine learning model trained on the dataset.
    - best_scaler (sklearn.preprocessing.RobustScaler): The scaler used for feature scaling in the best model's pipeline.

    This function performs the following steps:
    1. Separates features and target variable from the input DataFrame.
    2. Splits the data into training and test sets.
    3. Defines a set of machine learning models to be trained and hyperparameter grids for tuning.
    4. Trains each model using K-Fold cross-validation and hyperparameter tuning.
    5. Evaluates each model's performance on the test set and records the results.
    6. Selects and returns the best-performing model based on test accuracy.
    7. Returns the scaler used for feature scaling in the best model's pipeline.

    Example usage:
    best_model, best_scaler = generate_model(df)
    print("Best Model:", best_model)
    print("Best Scaler:", best_scaler)
    """
    # Separate features (X) and target (y)
    X = df.drop(labels=['Upcoming_event','Record ID','Date of Procedure','Date of Blood Draw'], axis=1)
    y = df['Upcoming_event']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'XGBoost': XGBClassifier()
    }

    # Define hyperparameter grids for tuning (adjust as needed)
    param_grids = {
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300]
            # Add other hyperparameters as needed
        }
    }

    # Initialize results dictionary
    results = {}

    # Store the best model and scaler
    best_model = None
    best_scaler = None
    best_score = 0  # Initialize with a low score

    # Perform K-Fold cross-validation and hyperparameter tuning for each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        # Create a Pipeline with feature scaling using RobustScaler
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # Use RobustScaler for feature scaling
            ('classifier', model)
        ])

        # Initialize K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grids[model_name],
                                scoring='accuracy', cv=kf)

        # Fit the model and perform hyperparameter tuning
        grid_search.fit(X_train, y_train)

        # Get the best model and its parameters
        best_params = grid_search.best_params_

        # Evaluate the best model on the test set
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store results in the dictionary
        results[model_name] = {
            'Best Parameters': best_params,
            'Test Accuracy': accuracy,
            'Test F1 Score (Macro)': f1
        }

        # Check if this model performed better than the current best
        if accuracy > best_score:
            best_score = accuracy
            best_model = grid_search.best_estimator_
            best_scaler = pipeline.named_steps['scaler']

        print(f"{model_name} training complete. Test Accuracy: {accuracy:.4f}, Test F1 Score (Macro): {f1:.4f}")

    # Print results
    for model_name, result in results.items():
        print(f"Model: {model_name}")
        print(f"Best Parameters: {result['Best Parameters']}")
        print(f"Test Accuracy: {result['Test Accuracy']:.4f}")
        print(f"Test F1 Score (Macro): {result['Test F1 Score (Macro)']:.4f}")
        print()

    # Return the best model and scaler
    return best_model

# # Load your dataset
# data_path = "./data/Preprocessed_Data.xlsx"
# df = pd.read_excel(data_path)

# # Example usage:
# best_model, best_scaler = generate_model(df)
# print("Best Model:", best_model)
# print("Best Scaler:", best_scaler)
