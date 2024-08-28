import pandas as pd
from scipy.stats import zscore

def preprocess_data(df,is_handing_outliers):
    """General preprocessing logics"""
    print ("Data preprocessing started")

    # Drop invoice_num column
    df_pp = df.drop(['invoice_num'], axis=1)

    # Dropping possible records with missing values
    df_pp = df_pp.dropna()

    # Change the data type of date_id column
    df_pp['date_id'] = pd.to_datetime(df_pp['date_id'])

    # Handing outliers
    if (is_handing_outliers):
        df_pp = handle_outliers(df_pp)

    if is_missing_dates(df_pp):
        pass

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


def is_missing_dates(df):
    """Checking missing dates in the date field"""
    print ("Checking missing dates started")

    # Copying the DataFrame
    df_copy = df.copy()

    df_copy.set_index('date_id', inplace=True)

    # Step 1: Identify the start and end dates of the actual data
    start_date = df_copy.index.min()
    end_date = df_copy.index.max()

    print("DataFrame Index Range:")
    print(df_copy.index.min(), df_copy.index.max())

    # Step 2: Generate a complete range of dates
    complete_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Step 3: Find missing dates
    missing_dates = complete_date_range.difference(df_copy.index)

    print ("Checking missing dates ended")

    # Display missing dates
    if missing_dates.empty:
        print ("No missing dates")
        return False
    else:
        print("Missing Dates:")
        print(missing_dates)
        return True
    


    