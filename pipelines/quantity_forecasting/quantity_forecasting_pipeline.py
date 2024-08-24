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
from src.utils.sales_quantity_features import create_sales_quantity_features
from src.utils.time_features import create_time_features
from src.utils.item_features import create_item_features
from src.utils.outlet_features import create_outlet_features
from src.utils.feature_target_variable import define_features_and_target_variables
from src.utils.save_util import save_model
from src.models.train_model import train_model
from src.models.evaluate_model import test_model


# from models.train_model import train_model
from conf.config_loader import get_config

def main():

    print ("Quantity forcasting pipeline started")
    
    # Loading the configuration
    config = get_config('conf/config.yaml')
    
    # Load datasets from config
    train_dataset = config['data']['train_dataset_path']
    test_dataset = config['data']['test_dataset_path']
    store_dataset = config['data']['store_dataset_path']
    
    train_df = load_data(train_dataset)
    test_df = load_data(test_dataset)
    store_df = load_data(store_dataset)

    print(train_df.info)
    print(test_df.info)

    # Preprocess Data
    train_pp_df = preprocess_data(train_df,True) # Outlier Handling is enabled
    test_pp_df = preprocess_data(test_df,False) # Outlier Handling is disabled
    
    # Set primary keys
    train_pp_df = set_primary_keys(train_pp_df)
    test_pp_df = set_primary_keys(test_pp_df)
    
    # Create sales quantity features
    train_pp_df = create_sales_quantity_features(train_pp_df)
    test_pp_df = create_sales_quantity_features(test_pp_df)

    # Create time features
    train_pp_df = create_time_features(train_pp_df)
    test_pp_df = create_time_features(test_pp_df)

    # Create item (department level) features
    train_pp_df,test_pp_df = create_item_features(train_pp_df,test_pp_df)

    # Create outlet (store) features
    train_pp_df,test_pp_df = create_outlet_features(train_pp_df,test_pp_df)


    print(train_pp_df.head(5))
    print(test_pp_df.head(5))

    # Define features and target variables
    train_features,train_target = define_features_and_target_variables(train_pp_df)
    test_features,test_target = define_features_and_target_variables(test_pp_df)
    
    # Train model
    model = train_model(train_features, train_target, config['model'])

    # Test model
    test_model(model,test_features,test_target)
    
    # Save model
    model_output_path = config['model']['output_path']
    save_model(model,model_output_path)

    print ("Quantity forcasting pipeline ended")

if __name__ == "__main__":
    main()