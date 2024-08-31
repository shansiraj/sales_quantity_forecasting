from src.utils.save_util import save_avg_feature_values

def setting_avg_feature_values(input_data,output_path):
    """Setting average values for selected features"""
    print ("Feature value setting started")

    columns_to_average = ['item_count', 'store_sales_ratio', 'revenue_per_item', 'net_sales', 'rolling_mean_7', 'lag_1']
    avg_summary_data = input_data[columns_to_average].mean()

    print("Average values of features:")
    print (avg_summary_data)

    save_avg_feature_values(avg_summary_data,output_path)

    print ("Feature value setting ended")

    return avg_summary_data