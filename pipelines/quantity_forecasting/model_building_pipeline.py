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
from src.models.evaluate_model import test_model
from src.models.model_selection import select_model,create_ml_model
from src.models.hyperparameter_tuning import optimize_hyperparameter
from src.models.feature_importance import analyze_feature_importance
from conf.config_loader import get_config

def pipeline(train_features,train_target,test_features,test_target):

    print ("Model building pipeline started")
    
    # Loading the configuration
    config = get_config('conf/config.yaml')
    
    # Model Selection
    is_model_selection = config['model']['is_model_selection']
    model = select_model(train_features,train_target,test_features,test_target,is_model_selection)

    is_hp_tuning = config['model']['is_hp_tuning']
    if (is_hp_tuning):
        # Hyper-parameter Tuning
        beset_params = optimize_hyperparameter(model,train_features,train_target)

        # Create final ML model
        model = create_ml_model(model,beset_params)

    # Train model
    model = train_model(model,train_features, train_target)

    is_feature_importance_analysis = config['model']['is_feature_importance_analysis']
    if (is_feature_importance_analysis):
        analyze_feature_importance(model,train_features)

    # Test model
    is_model_evaluation = config['model']['is_model_evaluation']
    if(is_model_evaluation):
        test_model(model,test_features,test_target)

    # Save model
    # model_output_path = config['model']['output_path']
    # save_model(model,model_output_path)

    print(train_features.head(5))

    print ("Model building pipeline ended")
