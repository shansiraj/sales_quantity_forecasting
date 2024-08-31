from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def test_model(model,X_test,y_test):

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print (f"Validation MAPE: {mape:.2f}") # Mean Absolute Percentage Error (MAPE): Percentage of error relative to the actual value.
    print (f"Validation MAE: {mae:.2f}") # Mean Absolute Error (MAE): Average absolute error between predicted and actual values.
    print (f"Validation RMSE: {rmse:.2f}") # Root Mean Squared Error (RMSE): Square root of MSE, giving error in the same unit as the target.
    print (f"Validation MSE: {mse:.2f}") # Mean Squared Error (MSE): Average squared error, penalizing larger errors more.
    print (f"Validation R2: {r2:.2f}") # R-squared: Measure of how well the model explains the variance in the data.
    
    store_wise_mape = calculate_store_wise_mape(X_test,y_test,y_pred)
    store_wise_department_wise_mape = calculate_store_wise_group_wise_mape(X_test,y_test,y_pred)

    print ("MAPE on Date and Store :")
    print (store_wise_mape)
    print()
    print ("MAPE on Date, Store and Department :")
    print (store_wise_department_wise_mape )

    store_wise_mape.to_csv('src/models/store_wise_mape.csv', index=False)
    store_wise_department_wise_mape.to_csv('src/models/store_wise_department_wise_mape.csv', index=False)

    evaluation_matrics = {'MAPE':mape,
                        'MAE':mae,
                        'RMSE': rmse,
                        'MSE': mse,
                        'R2': r2,
                        }
    
    return evaluation_matrics


def calculate_store_wise_mape(x_test,y_test, y_pred):

    y_test = pd.Series(y_test, name='y_test')
    y_pred = pd.Series(y_pred, name='y_pred')

    # Combine x_test with y_test and y_pred
    result_df = pd.concat([x_test, y_test, y_pred], axis=1)

    
    mape = result_df.groupby(['year','month','day','store_ABC', 'store_XYZ']).apply(
        lambda x: mean_absolute_percentage_error(x['y_test'], x['y_pred'])
    ).reset_index(name='mape')

    return mape

def calculate_store_wise_group_wise_mape(x_test,y_test, y_pred):

    y_test = pd.Series(y_test, name='y_test')
    y_pred = pd.Series(y_pred, name='y_pred')

    # Combine x_test with y_test and y_pred
    result_df = pd.concat([x_test, y_test, y_pred], axis=1)

    mape = result_df.groupby(['year','month','day','store_ABC', 'store_XYZ','item_dept_Beverages','item_dept_Grocery','item_dept_Household']).apply(
        lambda x: mean_absolute_percentage_error(x['y_test'], x['y_pred'])
    ).reset_index(name='mape')

    return mape
