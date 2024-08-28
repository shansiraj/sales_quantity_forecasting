
from conf.config_loader import get_config
from src.utils.data_loader import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_preprocessing_for_eda(train_df):
    train_df['date_id'] = pd.to_datetime(train_df['date_id'], errors='coerce')

    # Drop 'invoice_num' column
    train_df = train_df.drop(['invoice_num'], axis=1)

    # Convert categorical columns to category type
    train_df['item_dept'] = train_df['item_dept'].astype('category')
    train_df['store'] = train_df['store'].astype('category')
    train_df['item'] = train_df['item'].astype('category')

    return train_df

def basic_info(train_df,test_df,store_df):
    # Understanding the Structure of the datasets
    print("Shape of the Train Dataset \t\t :",train_df.shape)
    print("Shape of the Test Dataset \t\t :",test_df.shape)
    print("Shape of the Outlet Info Dataset \t :",store_df.shape)
    print("")

    # Checking the details of the dataset
    print("Details of the Train Dataset \t\t\n :",train_df.info())
    print("Details of the Test Dataset \t\t\n :",test_df.info())
    print("Details of the Outlet Info Dataset \t\n :",store_df.info())
    print("")

    # Check for missing values
    print("Null values in Training Dataset")
    print(train_df.isnull().sum())
    print("")

    print("Null values in Testing Dataset")
    print(test_df.isnull().sum())
    print("")

def statistical_info(train_df):
    print("Statistical Details")
    print(train_df.describe())
    print("")

def relationship_between_net_sales_and_itm_qty(train_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='net_sales', y='item_qty', data=train_df)
    plt.title('Scatter Plot of Net Sales vs. Item Quantity ')
    plt.xlabel('Net Sales')
    plt.ylabel('Item Quantity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.regplot(x='net_sales', y='item_qty', data=train_df,scatter_kws={'s':50}, line_kws={'color':'red'})
    plt.title('Scatter Plot of Net Sales vs. Item Quantity ')
    plt.xlabel('Net Sales')
    plt.ylabel('Item Quantity')
    plt.show()

def item_qty_by_store(train_df):
    total_quantities_by_dept = train_df.groupby('store')['item_qty'].sum().reset_index()
    total_quantities_by_dept = total_quantities_by_dept.sort_values(by='item_qty', ascending=False)

    print(total_quantities_by_dept)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='store', y='item_qty', data=total_quantities_by_dept, palette='viridis')
    plt.title('Total Quantities by Store')
    plt.xlabel('Store')
    plt.ylabel('Total Quantity')
    plt.xticks(rotation=45)
    plt.show()

def item_qty_by_department(train_df):
    total_quantities_by_dept = train_df.groupby('item_dept')['item_qty'].sum().reset_index()
    total_quantities_by_dept = total_quantities_by_dept.sort_values(by='item_qty', ascending=False)

    print(total_quantities_by_dept)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='item_dept', y='item_qty', data=total_quantities_by_dept, palette='viridis')
    plt.title('Total Quantities by Department')
    plt.xlabel('Department')
    plt.ylabel('Total Quantity')
    plt.xticks(rotation=45)
    plt.show()

def qty_by_store_and_department(train_df):

    # Create a FacetGrid for store-wise department plots
    g = sns.FacetGrid(train_df, col='store', col_wrap=2, height=5, sharey=False)

    # Map barplot to each facet
    g.map(sns.barplot, 'item_dept', 'item_qty', order=train_df['item_dept'].unique(), palette='viridis')

    # Adjust titles and labels
    g.set_titles('Store: {col_name}')
    g.set_axis_labels('Department', 'Total Quantity')
    g.set_xticklabels(rotation=45)
    g.add_legend()

    plt.tight_layout()
    plt.show()


def item_qty_by_date_for_store(train_df):
    quantity_by_date_dept = train_df.groupby(['date_id', 'store'])['item_qty'].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=quantity_by_date_dept, x='date_id', y='item_qty', hue='store', marker='o')
    plt.title('Total Quantity by Date for Each Store')
    plt.xlabel('Date')
    plt.ylabel('Total Quantity')
    plt.legend(title='Store')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def item_qty_by_date_for_department(train_df):
    quantity_by_date_dept = train_df.groupby(['date_id', 'item_dept'])['item_qty'].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=quantity_by_date_dept, x='date_id', y='item_qty', hue='item_dept', marker='o')
    plt.title('Total Quantity by Date for Each Department')
    plt.xlabel('Date')
    plt.ylabel('Total Quantity')
    plt.legend(title='Department')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def item_qty_by_date_for_store_and_department(train_df):

    quantity_by_date_store_dept = train_df.groupby(['date_id', 'store', 'item_dept'])['item_qty'].sum().reset_index()


    stores = quantity_by_date_store_dept['store'].unique()

    # Plot for each store
    for store in stores:
        # Filter data for the specific store
        filtered_data = quantity_by_date_store_dept[quantity_by_date_store_dept['store'] == store]

        # Create the plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=filtered_data, x='date_id', y='item_qty', hue='item_dept', marker='o')
        plt.title(f'Total Quantity by Date for Each Department - Store: {store}')
        plt.xlabel('Date')
        plt.ylabel('Total Quantity')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(title='Department')
        
        # Save or show the plot
        plt.tight_layout()
        plt.show()

def main():

    print ("EDA started")
    
    config = get_config('conf/config.yaml')
    
    # Load datasets from config
    train_dataset = config['data']['train_dataset_path']
    test_dataset = config['data']['test_dataset_path']
    store_dataset = config['data']['store_dataset_path']
    
    train_df = load_data(train_dataset)
    test_df = load_data(test_dataset)
    store_df = load_data(store_dataset)

    basic_info(train_df,test_df,store_df)
    statistical_info(train_df)
    train_df = data_preprocessing_for_eda(train_df)
    relationship_between_net_sales_and_itm_qty(train_df)
    item_qty_by_store(train_df)
    item_qty_by_department(train_df)
    qty_by_store_and_department(train_df)
    item_qty_by_date_for_store(train_df)
    item_qty_by_date_for_department(train_df)
    item_qty_by_date_for_store_and_department(train_df)

    print ("EDA ended")

if __name__ == "__main__":
    main()