# Import necessary libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Function to create a Gradient Boosting model
def create_gradient_boosting_model(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0):
    """
    Create a Gradient Boosting Regression model.

    Parameters:
        - X_train (pd.DataFrame or np.array): Training feature matrix.
        - y_train (pd.Series or np.array): Target values for training.
        - n_estimators (int): The number of boosting stages (default=100).
        - learning_rate (float): The step size shrinkage to prevent overfitting (default=0.1).
        - max_depth (int): Maximum depth of individual trees (default=3).
        - random_state (int): Seed for random number generation (default=0).

    Returns:
        - model: The trained Gradient Boosting model.
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

# Function to create a Random Forest model
def create_random_forest_model(X_train, y_train, n_estimators=100, max_depth=None, random_state=0):
    """
    Create a Random Forest Regression model.

    Parameters:
        - X_train (pd.DataFrame or np.array): Training feature matrix.
        - y_train (pd.Series or np.array): Target values for training.
        - n_estimators (int): The number of trees in the forest (default=100).
        - max_depth (int): Maximum depth of individual trees (default=None).
        - random_state (int): Seed for random number generation (default=0).

    Returns:
        - model: The trained Random Forest model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

# Function to create an ElasticNet model
def create_elasticnet_model(X_train, y_train, alpha=1.0, l1_ratio=0.5, random_state=None):
    """
    Create an ElasticNet Regression model.

    Parameters:
        - X_train (pd.DataFrame or np.array): Training feature matrix.
        - y_train (pd.Series or np.array): Target values for training.
        - alpha (float): Regularization strength (default=1.0).
        - l1_ratio (float): Mixing parameter for L1 (Lasso) and L2 (Ridge) regularization (default=0.5).
        - random_state (int): Seed for random number generation (default=None).

    Returns:
        - model: The trained ElasticNet model.
    """
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
