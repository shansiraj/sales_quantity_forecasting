def create_sales_quantity_features(df):
    """Create features related to sales quantity."""

    print ("Creating sales quantity related features started")

    # df['lag_1'] = df['item_qty'].shift(1)
    # df['rolling_mean_7'] = df['item_qty'].shift(1).rolling(window=7).mean()

    # Dropping possible missing values after aggregation
    df = df.dropna()

    print ("Creating sales quantity related features ended")

    return df