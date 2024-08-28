
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.utils.save_util import save_feature_scaler

def apply_feature_scaling(train_df,test_df,save_path):
    """Feature Scaling Logics"""
    print ("Feature scaling started")

    scaler = StandardScaler()
    scaler.fit(train_df)

    train_normalized = scaler.transform(train_df)
    test_normalized = scaler.transform(test_df)

    train_normalized_df = pd.DataFrame(train_normalized, columns=train_df.columns)
    test_normalized_df = pd.DataFrame(test_normalized, columns=test_df.columns)

    save_feature_scaler(scaler,save_path)

    print ("Feature scaling started")

    return train_normalized_df,test_normalized_df

def feature_scaling_for_input(scaler,input_data):
    """Feature Scaling Logics"""
    print ("Feature scaling started")

    input_normalized = scaler.transform(input_data)

    input_normalized_df = pd.DataFrame(input_normalized, columns=input_data.columns)

    print ("Feature scaling started")

    return input_normalized_df

