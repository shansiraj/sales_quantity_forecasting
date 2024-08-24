import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import joblib
import lightgbm as lgb



def init_model(model_config):
    model = RandomForestRegressor(
        n_estimators=model_config['n_estimators'],
        max_depth=model_config['max_depth'],
        random_state=model_config['random_state']
    )

    # model = lgb.LGBMRegressor()
    

    return model

def train_model(X_train, y_train, model_config):
    """
    Trains a RandomForestRegressor model.
    """
    
    # Initialize model
    model = init_model(model_config)

    print ("Model training started")
    
    # Train model
    model.fit(X_train, y_train)

    print ("Model training ended")
    
    # Validate model
    
    return model


def save_model(model, output_path):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, output_path)
    logging.info(f"Model saved to {output_path}")