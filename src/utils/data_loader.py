import pandas as pd

def load_data(file_path):
    """Loads the data from a CSV or other formats."""
    print ("The file ",file_path," loaded")
    df = pd.read_csv(file_path)
    return df