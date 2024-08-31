import joblib

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def load_scaler(filename):
    scaler = joblib.load(filename)
    print(f"Scaler loaded from {filename}")
    return scaler

def load_avg_feature_values(filename):
    avg_fuature_values = joblib.load(filename)
    print(f"Average feature values loaded from {filename}")
    return avg_fuature_values