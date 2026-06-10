import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_col='target'):
    # Simple preprocessing - adjust based on dataset
    if 'target' not in df.columns and target_col != 'target':
        df = df.rename(columns={df.columns[-1]: 'target'})
        target_col = 'target'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, dataset_name):
    os.makedirs('data/processed', exist_ok=True)
    # You can save as .npy or .pkl if needed