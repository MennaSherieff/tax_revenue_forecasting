import json
import pickle
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_all_models():
    scaler = load_pickle('models/scaler.pkl')
    lasso = load_pickle('models/lasso_model.pkl')
    ridge = load_pickle('models/ridge_model.pkl')
    xgb = load_pickle('models/xgb_model.pkl')
    return scaler, lasso, ridge, xgb

def load_data():
    # if you have CSVs or Excel in data/
    # return pd.read_csv('data/your_dataset.csv')
    pass
