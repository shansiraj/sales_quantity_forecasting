import pandas as pd
import matplotlib.pyplot as plt

def analyze_feature_importance(model,train_features):

    model_name = type(model).__name__

    if model_name == 'RandomForestRegressor':
        importances = model.feature_importances_
    elif model_name == 'LGBMRegressor':
        importances = model.feature_importances_
    elif model_name == 'CatBoostRegressor':
        importances = model.get_feature_importance()
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    # Create a DataFrame for easier plotting
    features = train_features.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance from LGBMRegressor')
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.show()