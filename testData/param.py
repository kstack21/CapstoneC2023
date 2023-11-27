param_grids = {
    'Gradient Boosting': {
        #  The number of boosting stages (trees) to be used in the ensemble.
        'classifier__n_estimators': [100, 200, 300],
        # The maximum depth of each tree in the ensemble.
        'classifier__max_depth': [3, 4, 5],
        # The step size shrinkage to reduce overfitting.
        'classifier__learning_rate': [0.1, 0.01, 0.001],
        # The minimum number of samples required to split an internal node.
        'classifier__min_samples_split': [2, 5, 10],
        # The minimum number of samples required to be in a leaf node.
        'classifier__min_samples_leaf': [1, 2, 4],
        # The number of features to consider when looking for the best split.
        'classifier__max_features': ['auto', 'sqrt', 'log2']
    },
    'Random Forest': {
        # The number of decision trees in the forest.
        'classifier__n_estimators': [100, 200, 300],
        # The maximum depth of each decision tree in the forest.
        'classifier__max_depth': [None, 10, 20],
        # The minimum number of samples required to split an internal node.
        'classifier__min_samples_split': [2, 5, 10],
        # The minimum number of samples required to be in a leaf node.
        'classifier__min_samples_leaf': [1, 2, 4],
        # The number of features to consider when looking for the best split.
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        # Whether to use bootstrap samples when building trees.
        'classifier__bootstrap': [True, False]
    },
    'XGBoost': {
        # The number of boosting rounds (trees) to be used in the ensemble.
        'classifier__n_estimators': [100, 200, 300],
        # The maximum depth of each tree in the ensemble.
        'classifier__max_depth': [3, 4, 5],
        # The step size shrinkage to reduce overfitting.
        'classifier__learning_rate': [0.1, 0.01, 0.001],
        # The fraction of samples used for training each tree.
        'classifier__subsample': [0.8, 0.9, 1.0],
        # The fraction of features used for training each tree.
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        # Minimum loss reduction required to make a further partition on a leaf node.
        'classifier__gamma': [0, 0.1, 0.2],
        # Minimum sum of instance weight (hessian) needed in a child.
        'classifier__min_child_weight': [1, 2, 5],
        # L1 regularization term on weights.
        'classifier__alpha': [0, 0.1, 1],
        #  L2 regularization term on weights.
        'classifier__lambda': [0, 0.1, 1]
    }
}
