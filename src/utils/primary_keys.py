def set_primary_keys(df):
    """Set the primary keys for the DataFrame."""
    print ("Setting primary keys started")

    # Aggregate the Sales Quantity ans Net Sales to get unique records for 'date_id', 'item_dept', 'store'
    df = df.groupby(['date_id', 'item_dept', 'store']).agg({'item_qty': 'sum','net_sales': 'sum','item':'count'}).reset_index()
    df.rename(columns={'item': 'item_count'}, inplace=True)

    # Sorting the dataset by date
    df.sort_values("date_id", inplace=True)

    print ("Setting primary keys ended")

    return df