import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from conf.config_loader import get_config
from src.utils.load_util import load_model
from src.utils.time_features import create_time_features
import plotly.express as px

def load_model_for_web_app():
    config = get_config('conf/config.yaml')
    # load model
    model_input_path = config['model']['output_path']
    model = load_model(model_input_path)

    return model

def generate_input_data(start_date,end_date,selected_department,selected_outlet):

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
                            'store_XYZ']]
    
    return input_data


def generate_output_data(start_date,end_date,selected_department,selected_outlet,predictions):

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')


    output_data_dic = {
            'Date': pd.date_range(start=start_date_str, end=end_date_str, freq='D').date,
            'Department': [selected_department] * ((end_date - start_date).days + 1),
            'Outlet': [selected_outlet] * ((end_date - start_date).days + 1),
            'Predicted Sales': [None] * ((end_date - start_date).days + 1)
        }
    
    output_data = pd.DataFrame(output_data_dic)

    output_data['Predicted Sales'] = predictions

    
    return output_data


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
        <p>Developed by D. Shan Siraj </br>
        STNO: COMSCDS231P-023</p>
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
            # Set the 'Date' column as the index for time series plotting
            # output_data.set_index('Date', inplace=True)
            # st.write("Predicted Sales Line Chart")
            # st.line_chart(output_data['Predicted Sales'])

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

    format_page()

if __name__ == "__main__":
    main()

