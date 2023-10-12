import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


# Load your dataset
data_path = "./data/Preprocessed_Data.xlsx"
df = pd.read_excel(data_path)


# Separate features (X) and target (y)
X = df.drop(labels=['Upcoming_event','Record ID','Date of Procedure','Date of Blood Draw'], axis=1)
y = df['Upcoming_event']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

# Define hyperparameter grids for tuning (adjust as needed)
param_grids = {
    'Random Forest': {
        # The number of decision trees in the forest.
        'classifier__n_estimators': [100, 200, 300],
        # # The maximum depth of each decision tree in the forest.
        # 'classifier__max_depth': [None, 10, 20],
        # # The minimum number of samples required to split an internal node.
        # 'classifier__min_samples_split': [2, 5, 10],
        # # The minimum number of samples required to be in a leaf node.
        # 'classifier__min_samples_leaf': [1, 2, 4],
        # # The number of features to consider when looking for the best split.
        # 'classifier__max_features': ['auto', 'sqrt', 'log2'],
        # # Whether to use bootstrap samples when building trees.
        # 'classifier__bootstrap': [True, False]
    },
    'XGBoost': {
        # The number of boosting rounds (trees) to be used in the ensemble.
        'classifier__n_estimators': [100, 200, 300],
        # # The maximum depth of each tree in the ensemble.
        # 'classifier__max_depth': [3, 4, 5],
        # # The step size shrinkage to reduce overfitting.
        # 'classifier__learning_rate': [0.1, 0.01, 0.001],
        # # The fraction of samples used for training each tree.
        # 'classifier__subsample': [0.8, 0.9, 1.0],
        # # The fraction of features used for training each tree.
        # 'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        # # Minimum loss reduction required to make a further partition on a leaf node.
        # 'classifier__gamma': [0, 0.1, 0.2],
        # # Minimum sum of instance weight (hessian) needed in a child.
        # 'classifier__min_child_weight': [1, 2, 5],
        # # L1 regularization term on weights.
        # 'classifier__alpha': [0, 0.1, 1],
        # #  L2 regularization term on weights.
        # 'classifier__lambda': [0, 0.1, 1]
    }
}


# Initialize results dictionary
results = {}

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
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Store results in the dictionary
    results[model_name] = {
        'Best Model': best_model,
        'Best Parameters': best_params,
        'Test Accuracy': accuracy,
        'Test F1 Score (Macro)': f1
    }

    print(f"{model_name} training complete. Test Accuracy: {accuracy:.4f}, Test F1 Score (Macro): {f1:.4f}")

# Print results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Best Parameters: {result['Best Parameters']}")
    print(f"Test Accuracy: {result['Test Accuracy']:.4f}")
    print(f"Test F1 Score (Macro): {result['Test F1 Score (Macro)']:.4f}")
    print()
