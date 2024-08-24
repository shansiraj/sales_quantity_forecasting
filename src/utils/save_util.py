import joblib
import os

def save_model(model, filename):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def save_predictions(predictions, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    predictions.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def save_evaluation_metrics(metrics, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Evaluation metrics saved to {filename}")