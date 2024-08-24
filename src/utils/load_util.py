import joblib

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model