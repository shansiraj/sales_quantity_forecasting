from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def create_item_features(df):
    """Create item-related (deparment level) features."""

    print ("Creating item (deparment level) features started")

    df['revenue_per_item'] = df['net_sales'] / df['item_count']

    print ("Creating item (deparment level) features ended")

    return df