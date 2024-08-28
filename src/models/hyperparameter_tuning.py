
from sklearn.model_selection import GridSearchCV

def optimize_hyperparameter(model,train_features,train_target):

    model_name = type(model).__name__

    if model_name == 'RandomForestRegressor':
        
        # Hyperparameter grid for RandomForestRegressor
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        }
    
    elif model_name == 'LGBMRegressor':
        
        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of boosting iterations
            'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
            'num_leaves': [31, 50, 70],  # Number of leaves in one tree
            'max_depth': [-1, 10, 20],  # Maximum depth of the tree
            'min_child_samples': [20, 50, 100],  # Minimum number of data points in a leaf
            'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting the trees
            'colsample_bytree': [0.8, 0.9, 1.0]  # Fraction of features used for fitting each tree
        }
    
    elif model_name == 'CatBoostRegressor':
        
        # Define the parameter grid
        param_grid = {
            'iterations': [100, 200, 300],  # Number of boosting iterations
            'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
            'depth': [6, 8, 10],  # Depth of the trees
            'l2_leaf_reg': [1, 3, 5],  # L2 regularization coefficient
            'border_count': [32, 50, 100],  # Number of splits for numerical features
            'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting the trees
            'colsample_bylevel': [0.8, 0.9, 1.0]  # Fraction of features used for each split
        }
    
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(train_features, train_target)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    print(f"Best Parameters for {model_name}:", best_params)
    return best_params