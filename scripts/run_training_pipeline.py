import sys
import os
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Data.load import load_data
from src.Data.preprocessing import preprocess_data
from src.Model.model import dmpnn
from src.Model.tuining import tune_model
from src.Model.trainer import start_training

#Loading data
train, test, valid = load_data()

#preprocessing
train_loader, test_loader, valid_loader, target_scaler, molecular_features_size, extra_feats_scaler = preprocess_data(train, test, valid).run_pipeline()

#model tuining
best_value, params = tune_model(
    train_loader,
    valid_loader,
    target_scaler,
    mol_feat_size=molecular_features_size,
    epoches=10
    ).start_tuining()

model = dmpnn(
    scale = target_scaler,
    mol_feat_size=molecular_features_size,
    **params
)

trainer = start_training(model, train_loader, valid_loader)
print("\nTraining is completed test results are:-")
trainer.test(model=model,dataloaders=test_loader, weights_only=False)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("optuna_search")

with mlflow.start_run():
    mlflow.sklearn.log_model(sk_model=extra_feats_scaler, artifact_path="scaler")
    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")

