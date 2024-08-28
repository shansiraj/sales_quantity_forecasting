def select_features(df):

    df = df.drop(['store'], axis=1)
    df = df.drop(['item_dept'], axis=1)

    df = df[['year',
            'month',
            'day',
            'day_of_week',
            'is_weekend',
            'item_dept_Beverages',
            'item_dept_Grocery',
            'item_dept_Household',
            'store_ABC',
            'store_XYZ',
            'store_sales_ratio',
            'revenue_per_item',
            'net_sales',
            'lag_1',
            'rolling_mean_7',
            'item_qty']]
    
    return df
