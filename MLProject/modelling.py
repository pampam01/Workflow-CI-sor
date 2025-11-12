import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os 

# Set nama eksperimen MLflow
# MLflow akan membuat folder 'mlruns' untuk menyimpan data secara lokal
# mlflow.set_tracking_uri("http://127.0.0.1:5000") # Opsional, tapi disarankan# Opsional, tapi disarankan
mlflow.set_experiment("Stroke Prediction Basic")

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
    # Mengaktifkan autologging dari MLflow untuk Scikit-Learn
    # Ini akan secara otomatis mencatat parameter, metrik, dan artefak model
    mlflow.sklearn.autolog()

    # Tentukan path ke data bersih Anda
    DATA_PATH = "cleaned/healthcare-dataset-stroke-data_cleaned.csv"
    
    df = load_data(DATA_PATH)
    if df is None:
        return

    # Sesuai notebook Anda, 'stroke' adalah target
    target_column = 'stroke'
    
    # Pastikan 'stroke' ada di dataframe
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the data.")
        return

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into train and test sets.")

    # Memulai run MLflow
    with mlflow.start_run() as run:
        print(f"Starting MLflow run with ID: {run.info.run_id}")

        # Inisialisasi dan latih model sederhana (sesuai Kriteria Basic)
        # LogisticRegression sering digunakan untuk data stroke
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        print("Model training completed.")

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi (Meskipun autolog mencatat ini, baik untuk verifikasi)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Autolog akan menangani logging:
        # 1. Parameter (C, max_iter, dll.)
        # 2. Metrik (accuracy, precision, recall, f1)
        # 3. Artefak (model.pkl, confusion_matrix.png, dll.)
        
        print("MLflow autologging completed.")
        print(f"Run {run.info.run_id} finished.")

if __name__ == "__main__":
    main()
