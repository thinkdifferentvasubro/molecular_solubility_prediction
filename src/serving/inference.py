import os

import mlflow
import torch
from lightning import pytorch as pl

from src.serving.inference_preprocessing_pipeline import (
    preprocess_inference_data,
)

current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL = mlflow.pytorch.load_model(
    os.path.join(current_dir, "model")
)

SCALER = mlflow.sklearn.load_model(
    os.path.join(current_dir, "scaler")
)

TRAINER = pl.Trainer(
    logger=None,
    enable_progress_bar=False,
    accelerator="cpu",
    devices=1,
)


def predict_solubility(
    smiles,
    testing=False,
    test_model=None,
    test_scaler=None,
):
    if testing:
        model = test_model
        scaler = test_scaler
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=False,
            accelerator="cpu",
            devices=1,
        )
    else:
        model = MODEL
        scaler = SCALER
        trainer = TRAINER

    preprocessing = preprocess_inference_data(scaler)

    loader, valid_smiles = preprocessing.run_pipeline(smiles)

    with torch.inference_mode():
        test_preds = trainer.predict(model, loader)

    preds = torch.cat(test_preds).view(-1).tolist()
    
    return preds, valid_smiles