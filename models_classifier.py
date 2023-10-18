import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

def generate_model(df):
    """
    Generate and evaluate an XGBoost machine learning model with hyperparameter tuning.

    This function takes a pandas DataFrame as input, splits it into features and target, creates an XGBoost classifier,
    performs feature scaling using RobustScaler, and sets up a pipeline for hyperparameter tuning.
    It then conducts K-Fold cross-validation with randomized hyperparameter search to find the best model.

    Parameters:
    - df (pd.DataFrame): The input pandas DataFrame containing your dataset, including features and target.

    Returns:
    - best_model (sklearn.pipeline.Pipeline): The best machine learning model found after hyperparameter tuning.
    - best_params (dict): The best hyperparameters for the model.
    - accuracy (float): The accuracy score of the best model on a hold-out test set.

    Example usage:
    best_model, best_params, accuracy = generate_model(df)

    Note:
    1. Ensure that the target column is named 'Upcoming_event' in your input DataFrame.
    2. You can adjust the hyperparameter grid (param_dist) to suit your specific problem.

    Dependencies:
    - pandas, xgboost, sklearn.model_selection, sklearn.metrics, sklearn.preprocessing, and sklearn.pipeline.

    Reference:
    - The function uses RandomizedSearchCV for hyperparameter tuning with K-Fold cross-validation.

    """


    # Confirm with CDR, what new columns are we generating?
    # Separate features (X) and target (y)
    X = df.drop(labels=['Upcoming_event'], axis=1)
    y = df['Upcoming_event']

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
    param_dist = {
        # The maximum depth of each tree in the ensemble.
        'classifier__max_depth': [3, 4, 5],
        # Minimum loss reduction required to make a further partition on a leaf node.
        'classifier__gamma': [0, 0.1, 0.2],
        # Minimum sum of instance weight (hessian) needed in a child.
        'classifier__min_child_weight': [1, 2, 5]
    }

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize GridSearchCV for hyperparameter tuning
    # CONFIRM WITH CDR. 
    # -   Might not ue accuracy to score.
    # -   Choose n_iter 
    randomized_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,
                                           n_iter=25, scoring='f1', cv=kf, random_state=42)

    # Fit the model and perform hyperparameter tuning
    randomized_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    scoring = accuracy_score(y_test, y_pred)


    # CONFIRM WITH CDR. Accuracy might not be the one used to score, best params will also change maybe
    return best_model, best_params, scoring, X_train


if __name__ == "__main__":
    # Load your dataset
    data_path = "./data/Preprocessed_Data.xlsx"
    df = pd.read_excel(data_path)


    best_model, best_params, scoring, X_train = generate_model(df)
    print("Best Model:", best_model)
    print("Best Parameters:", best_params)
    print("Test F1:", scoring)
