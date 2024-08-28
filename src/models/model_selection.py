
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from src.models.train_model import train_model
from src.models.evaluate_model import test_model
import pandas as pd

def select_model(train_features,train_target,test_features,test_target,is_model_selection):

    if not is_model_selection:

        best_model = CatBoostRegressor(
                    iterations=100,          # Number of boosting iterations
                    learning_rate=0.01,      # Learning rate
                    depth=6,                 # Depth of the trees
                    l2_leaf_reg=3,           # L2 regularization coefficient
                    border_count=50,         # Number of splits for numerical features
                    subsample=0.9,           # Fraction of samples used for fitting the trees
                    colsample_bylevel=0.9    # Fraction of features used for each split
                )
        return best_model

    else:
        models = [RandomForestRegressor()
                ,lgb.LGBMRegressor()
                ,CatBoostRegressor()
                ]

        results = {}
        plt.figure(figsize=(8, 8))

        # Train and evaluate each model
        for model in models:
            model_name = type(model).__name__

            trained_model = train_model(model,train_features,train_target)
            evaluation_matrics = test_model(trained_model,test_features,test_target)

            results[model_name] = evaluation_matrics

        # Display the evaluation results
        results_df = pd.DataFrame(results).T
        print(results_df)

        # Choose the best model based on MAPE score
        best_model_name = results_df['MAPE'].idxmin()
        best_model = next(model for model in models if type(model).__name__ == best_model_name)

        print(f"\nBest Model: {best_model_name}")
    
    return best_model


def create_ml_model(model,best_params):

    model_name = type(model).__name__
    final_model = None

    if model_name == 'RandomForestRegressor':
        final_model = RandomForestRegressor(**best_params)
    
    elif model_name == 'LGBMRegressor':
        final_model = lgb.LGBMRegressor(**best_params)
    
    elif model_name == 'CatBoostRegressor':
        final_model = CatBoostRegressor(**best_params)
    
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    return final_model