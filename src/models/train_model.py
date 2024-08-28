def train_model(model,X_train, y_train):
    """
    Trains a RandomForestRegressor model.
    """

    print ("Model training started")
    
    # Train model
    model.fit(X_train, y_train)

    print ("Model training ended")
    
    return model