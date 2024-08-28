from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def create_outlet_features(df):
    """Create outlet(store) related features."""

    print ("Creating outlet(store) features started")

    df['store_sales_ratio'] = df['net_sales'] / df['item_qty']

    print ("Creating outlet(store) features ended")

    return df
