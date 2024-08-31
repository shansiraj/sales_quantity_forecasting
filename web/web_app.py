import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from conf.config_loader import get_config
from src.utils.load_util import load_model,load_scaler,load_avg_feature_values
from src.utils.time_features import create_time_features
from src.utils.feature_scaling import feature_scaling_for_input
import plotly.express as px
from src.utils.data_loader import load_data
from src.utils.data_processor import preprocess_data
from src.utils.primary_keys import set_primary_keys

def load_model_for_web_app():
    config = get_config('conf/config.yaml')
    # load model
    model_input_path = config['model']['output_path']
    model = load_model(model_input_path)

    return model

def generate_input_data(start_date,end_date,selected_department,selected_outlet):

    config = get_config('conf/config.yaml')
    avg_feature_input_path = config['data_prep']['avg_feature_output_path']
    avg_feature_values = load_avg_feature_values(avg_feature_input_path)

    net_sales = avg_feature_values['net_sales'] # 250355.130000 
    lag_1 = avg_feature_values['lag_1'] #883.000
    rolling_mean_7 = avg_feature_values['rolling_mean_7'] #1393.351286
    item_count = avg_feature_values['item_count'] #3
    store_sales_ratio = avg_feature_values['store_sales_ratio'] #0
    revenue_per_item = avg_feature_values['revenue_per_item'] #0

    print(avg_feature_values,'..........................')

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Transform department in to binary vector
    household = 1
    beverages = 0
    grocery = 0

    if(selected_department=='Household'):
        household = 1
        beverages = 0
        grocery = 0
    elif(selected_department=='Beverages'):
        household = 0
        beverages = 1
        grocery = 0
    elif(selected_department=='Grocery'):
        household = 0
        beverages = 0
        grocery = 1

    # Transform outlet in to binary vector
    store_abc = 1
    store_xyz = 0

    if(selected_outlet=='ABC'):
        store_abc = 1
        store_xyz = 0
    elif(selected_outlet=='XYZ'):
        store_abc = 0
        store_xyz = 1


    input_data_dic = {
            'date_id': pd.date_range(start=start_date_str, end=end_date_str, freq='D'),
            'item_dept_Beverages': [beverages] * ((end_date - start_date).days + 1),
            'item_dept_Grocery': [grocery] * ((end_date - start_date).days + 1),
            'item_dept_Household': [household] * ((end_date - start_date).days + 1),
            'store_ABC': [store_abc] * ((end_date - start_date).days + 1),
            'store_XYZ': [store_xyz] * ((end_date - start_date).days + 1),
            'net_sales':[net_sales] * ((end_date - start_date).days + 1),
            'lag_1':[lag_1] * ((end_date - start_date).days + 1),
            'rolling_mean_7':[rolling_mean_7] * ((end_date - start_date).days + 1),
            'store_sales_ratio':[store_sales_ratio] * ((end_date - start_date).days + 1),
            'revenue_per_item':[revenue_per_item] * ((end_date - start_date).days + 1),
            'item_count':[item_count] * ((end_date - start_date).days + 1),
        }
    
    input_data = pd.DataFrame(input_data_dic)
    input_data = create_time_features(input_data)
    input_data = input_data[['year',
                            'month',
                            'day',
                            'day_of_week',
                            'is_weekend',
                            'item_dept_Beverages',
                            'item_dept_Grocery',
                            'item_dept_Household',
                            'store_ABC',
                            'store_XYZ',
                            'item_count',
                            'store_sales_ratio',
                            'revenue_per_item',
                            'net_sales',
                            'lag_1',
                            'rolling_mean_7']]
    
    config = get_config('conf/config.yaml')

    is_feature_scaling = config['data_prep']['is_feature_scaling']

    if (is_feature_scaling):
        scaler_input_path = config['data_prep']['scaler_output_path']
        scaler = load_scaler(scaler_input_path)
        input_data = feature_scaling_for_input(scaler,input_data)
    
    return input_data


def generate_output_data(start_date,end_date,selected_department,selected_outlet,predictions):

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')


    output_data_dic = {
            'Date': pd.date_range(start=start_date_str, end=end_date_str, freq='D').date,
            'Outlet': [selected_outlet] * ((end_date - start_date).days + 1),
            'Department': [selected_department] * ((end_date - start_date).days + 1),
            'Predicted Sales': [None] * ((end_date - start_date).days + 1)
        }
    
    output_data = pd.DataFrame(output_data_dic)

    output_data['Predicted Sales'] = predictions

    
    return output_data

def generate_test_dataset_output_data(selected_department,selected_outlet):

    # Loading the configuration
    config = get_config('conf/config.yaml')
    test_dataset = config['data']['test_dataset_path']
    test_df = load_data(test_dataset)

    test_pp_df = preprocess_data(test_df,False) # Outlier Handling is disabled
    test_pp_df = set_primary_keys(test_pp_df)

    filtered_data = test_pp_df[
    (test_pp_df['store'] == selected_outlet) &
    (test_pp_df['item_dept'] == selected_department)
    ]

    filtered_data.sort_values("date_id", inplace=True)

    filtered_data['Date'] = filtered_data['date_id'].dt.date
    filtered_data = filtered_data.drop(['date_id'], axis=1)

    filtered_data['Actual Sales'] = filtered_data['item_qty']
    filtered_data = filtered_data.drop(['item_qty'], axis=1)
    
    return filtered_data


def generate_forecasting(input_data):

    model = load_model_for_web_app()
    predictions = model.predict(input_data)

    return predictions

def format_page():
    st.sidebar.markdown(
    """
    <style>
    .sidebar-footer {
        position: relative;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 10px;
        font-size: 12px;
        color: #666;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        margin-top: 20px;  /* Add margin to separate from other sidebar content */
    }
    </style>
    <div class="sidebar-footer">
        <p>Developed by D. Shan Siraj</p>
    </div>
    """,
    unsafe_allow_html=True
    )


def main():

    # Set the page layout to wide
    st.set_page_config(layout="wide")
    st.title('Sales Forecasting Dashboard')
    st.sidebar.title("Sales Forecasting App")
    st.sidebar.header('User Input')

    # Date selection (start to end)
    start_date = st.sidebar.date_input('Start Date', datetime(2022, 2, 1))
    end_date = st.sidebar.date_input('End Date', datetime(2022, 2, 28))

    # Department selection
    departments = ['Beverages', 'Grocery', 'Household']
    selected_department = st.sidebar.selectbox('Select Department', departments)

    # Outlet selection
    outlets = ['XYZ', 'ABC']
    selected_outlet = st.sidebar.selectbox('Select Outlet', outlets)

    # Display the selected inputs
    st.write(f'**Selected Date Range:** {start_date} to {end_date}')
    st.write(f'**Selected Department:** {selected_department}')
    st.write(f'**Selected Outlet:** {selected_outlet}')

    # Button to trigger the activity
    if st.sidebar.button('Generate Forecast'):

        # create the input dataset
        input_data = generate_input_data(start_date,end_date,selected_department,selected_outlet)

        # generate the predictions
        predictions = generate_forecasting(input_data)

        # create the input dataset
        output_data = generate_output_data(start_date,end_date,selected_department,selected_outlet,predictions)

        # Create two columns
        col1, col2 = st.columns([1, 2])

        # Display the DataFrame in the first column
        with col1:
            st.write("Data Overview")
            st.write(output_data)

        # Display the line chart in the second column
        with col2:

            fig = px.line(output_data, x='Date', y='Predicted Sales', title="Predicted Sales Line Chart")
    
            # Customize the layout to add grid lines
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Predicted Sales",
                xaxis=dict(showgrid=True),  # Show grid lines on x-axis
                yaxis=dict(showgrid=True),  # Show grid lines on y-axis
                title_x=0.5  # Center the title
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

    if st.sidebar.button('Evaluate Model'):

        test_data = generate_test_dataset_output_data(selected_department,selected_outlet)

        # create the input dataset
        input_data = generate_input_data(start_date,end_date,selected_department,selected_outlet)

        # generate the predictions
        predictions = generate_forecasting(input_data)

        # create the input dataset
        output_data = generate_output_data(start_date,end_date,selected_department,selected_outlet,predictions)

        merged_df = pd.merge(test_data, output_data, on='Date')

        merged_df = merged_df[['Date',
                            'Outlet',
                            'Department',
                            'Actual Sales',
                            'Predicted Sales']]

        # Create two columns
        col1, col2 = st.columns([1, 2])

        # Display the DataFrame in the first column
        with col1:
            st.write("Data Overview")
            st.write(merged_df)


        # Display the line chart in the second column
        with col2:

            fig = px.line(merged_df, x='Date', y=['Actual Sales', 'Predicted Sales'], title="Actual vs Predicted Sales")
    
            # Customize the layout to add grid lines
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Actual",
                xaxis=dict(showgrid=True),  # Show grid lines on x-axis
                yaxis=dict(showgrid=True),  # Show grid lines on y-axis
                title_x=0.5  # Center the title
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

    format_page()

if __name__ == "__main__":
    main()

