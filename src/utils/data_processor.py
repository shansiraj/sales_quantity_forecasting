import pandas as pd
from scipy.stats import zscore

def preprocess_data(df,is_handing_outliers):
    """General preprocessing logics"""
    print ("Data preprocessing started")

    # Drop invoice_num column
    df_pp = df.drop(['invoice_num'], axis=1)

    # Change the data type of date_id column
    df_pp['date_id'] = pd.to_datetime(df_pp['date_id'])

    # Aggregate the Sales Quantity to generate summary report
    df_pp = df_pp.groupby(['date_id', 'item_dept', 'store'])['item_qty'].sum().reset_index()

    # Dropping possible missing values after aggregation
    df_pp = df_pp.dropna()

    # Sorting the dataset by date
    df_pp.sort_values("date_id", inplace=True)

    # Handing outliers
    if (is_handing_outliers):
        df_pp = handle_outliers(df_pp)

    print ("Data preprocessing ended")

    return df_pp

def handle_outliers(df):
    """Handling outliers"""
    print ("Outlier handling started")

    df['item_qty_zscore'] = zscore(df['item_qty'])

    # Removing outliers
    threshold = 3
    # Threshold Z-score > 3 or < -3
    # Filter out the rows where the Z-score > threshold or Z-score < threshold
    df_pp = df[(df['item_qty_zscore'] <= threshold) & (df['item_qty_zscore'] >= -threshold)]

    # Drop the z-score column
    df_pp = df_pp.drop(columns=['item_qty_zscore'])

    print ("Outlier handling ended")

    return df_pp