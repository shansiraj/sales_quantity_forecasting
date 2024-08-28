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
    
    evaluation_matrics = {'MAPE':mape,
                        'MAE':mae,
                        'RMSE': rmse,
                        'MSE': mse,
                        'R2': r2,
                        }
    
    return evaluation_matrics


def display_model_evaluation_plot(x_test_without_preprocess,model,X_test,y_test):

    y_pred = model.predict(X_test)

    print (x_test_without_preprocess)

    # df_deep_copy['year'] = pd.to_numeric(df_deep_copy['year'], errors='coerce')
    # df_deep_copy['month'] = pd.to_numeric(df_deep_copy['month'], errors='coerce')
    # df_deep_copy['day'] = pd.to_numeric(df_deep_copy['day'], errors='coerce')

    # # Create a 'date' column
    # df_deep_copy['date'] = pd.to_datetime(df_deep_copy[[ 'day','month','year']])

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plot actual values
    plt.plot(x_test_without_preprocess['date_id'], y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=4)

    # Plot predicted values
    # plt.plot(x_test_without_preprocess['date_id'],y_pred, label='Predicted', color='red', marker='x', linestyle='--', markersize=4)


    # # Plot actual values
    # plt.plot(range(len(y_test)), y_test, label='Actual', color='blue', marker='o')

    # # Plot predicted values
    # plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', marker='x')

    # Adding labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

