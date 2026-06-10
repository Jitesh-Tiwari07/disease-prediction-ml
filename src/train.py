import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

def preprocess_data(df, target_col):
    """Clean data properly"""
    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Handle target column
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    
    # Handle remaining missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def train_for_dataset(dataset_name, file_path):
    print(f"\n🔄 Training models for {dataset_name}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Detect target column
    possible_targets = ['target', 'diagnosis', 'Outcome', 'class']
    target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
    
    # Preprocess
    df = preprocess_data(df, target_col)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    models = get_models()
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            model_path = f'models/{dataset_name}_{name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model, model_path)
            print(f"   ✅ {name}: {acc:.4f} accuracy")
        except Exception as e:
            print(f"   ❌ {name} failed: {str(e)[:100]}...")
    
    print(f"✅ Completed training for {dataset_name}\n")

if __name__ == "__main__":
    print("=== Disease Prediction Training Started ===\n")
    
    datasets = {
        "heart": "data/raw/heart_disease.csv",
        "diabetes": "data/raw/diabetes.csv",
        "breast_cancer": "data/raw/breast_cancer.csv"
    }
    
    for name, path in datasets.items():
        train_for_dataset(name, path)
    
    print("🎉 All training completed! Models saved in 'models/' folder.")