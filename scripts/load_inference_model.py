import mlflow
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

download_path = os.path.join(BASE_DIR, "src", "serving")

run_id = "e9c0c8e0a3e5402eb95c396ec4d827e1"

# 🔹 download model
model_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model",
    dst_path=download_path
)

# 🔹 download scaler
scaler_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/scaler",
    dst_path=download_path
)

print("download completed")