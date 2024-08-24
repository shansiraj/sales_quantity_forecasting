from sklearn.metrics import mean_absolute_error, mean_squared_error

def test_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print (f"Validation MAE: {mae:.2f}")
    print (f"Validation RMSE: {rmse:.2f}")