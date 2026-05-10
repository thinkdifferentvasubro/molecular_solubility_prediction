import os
import mlflow
from mlflow.artifacts import download_artifacts


def loading_and_saving(
        tracking_uri="http://127.0.0.1:5000",
        model_uri="runs:/e9c0c8e0a3e5402eb95c396ec4d827e1/model",
        scaler_uri="runs:/e9c0c8e0a3e5402eb95c396ec4d827e1/scaler"
        ):
    MLFLOW_TRACKING_URI = tracking_uri

    MODEL_URI = model_uri
    SCALER_URI = scaler_uri

    download_path = os.path.abspath(os.path.join("src", "serving"))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Downloading model artifacts from MLflow...")
    model_path = download_artifacts(
        artifact_uri=MODEL_URI,
        dst_path=download_path
    )
    print(f"Model downloaded successfully!")
    print(f"Saved at: {model_path}\n")

    print("Downloading scaler artifacts from MLflow...")
    scaler_path = download_artifacts(
        artifact_uri=SCALER_URI,
        dst_path=download_path
    )
    print(f"scaler downloaded successfully!")
    print(f"Saved at: {scaler_path}")