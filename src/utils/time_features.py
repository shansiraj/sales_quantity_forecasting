def create_time_features(df):
    """Create time-related features."""

    print ("Creating time related features started")

    df['year'] = df['date_id'].dt.year
    df['month'] = df['date_id'].dt.month
    df['day'] = df['date_id'].dt.day
    df['day_of_week'] = df['date_id'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df.drop(columns=['date_id'], inplace=True)

    print ("Creating time related features ended")

    return df