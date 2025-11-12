import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os 

# DIUBAH: Pastikan baris ini AKTIF (TIDAK DI-KOMENTAR)
# Ini agar skrip terhubung ke server yang Anda jalankan di Langkah 1
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# DIHAPUS: Baris ini dihapus untuk menghindari konflik
# mlflow.set_experiment("Stroke Prediction Basic") 

def load_data(path):
    """Memuat data bersih."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None

def main():
    # Mengaktifkan autologging. 
    # Ini akan otomatis log ke run yang dibuat oleh 'mlflow run'
    mlflow.sklearn.autolog()

    DATA_PATH = "cleaned/healthcare-dataset-stroke-data_cleaned.csv"
    
    df = load_data(DATA_PATH)
    if df is None:
        return

    target_column = 'stroke'
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the data.")
        return

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into train and test sets.")

    # DIHAPUS: Blok 'with mlflow.start_run()' dihapus.
    # 'mlflow run' sudah mengaturnya.
    
    # Pastikan kode di bawah ini TIDAK menjorok (tidak di dalam 'with')
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    print("MLflow autologging completed.")

if __name__ == "__main__":
    main()
