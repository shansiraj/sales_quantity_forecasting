import pandas as pd

import os
import sys
import logging
import pandas as pd

# Add src directory to path (optional)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.data_loader import load_data
from src.utils.data_processor import preprocess_data
from src.utils.primary_keys import set_primary_keys
from src.utils.sales_features import create_sales_quantity_features
from src.utils.time_features import create_time_features
from src.utils.item_features import create_item_features
from src.utils.outlet_features import create_outlet_features
from src.utils.feature_scaling import apply_feature_scaling
from src.utils.feature_target_variable import define_features_and_target_variables
from src.utils.save_util import save_model
from src.models.train_model import train_model
from src.models.evaluate_model import test_model, display_model_evaluation_plot
from src.models.model_selection import select_model,create_ml_model
from src.models.hyperparameter_tuning import optimize_hyperparameter
from src.models.feature_importance import analyze_feature_importance


# from models.train_model import train_model
from pipelines.quantity_forecasting import data_preparation_pipeline
from pipelines.quantity_forecasting import model_building_pipeline

def main():

    print ("Main pipeline started")
    
    train_features,train_target,test_features,test_target = data_preparation_pipeline.pipeline()
    model_building_pipeline.pipeline(train_features,train_target,test_features,test_target)
   
    print ("Main pipeline ended")

if __name__ == "__main__":
    main()