from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.serving.prediction import predict


# ================================
# FASTAPI APP
# ================================
app = FastAPI(
    title="Molecular Solubility Prediction",
    description="ML API for predicting solubility of molecules",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}


class Smile(BaseModel):
    smiles: List[str]


@app.post("/predict")
def get_prediction(data: Smile):
    result = predict(data.smiles)
    return result


# ================================
# GRADIO FUNCTION
# ================================
def gradio_interface(smiles: str):
    smiles_list = [s.strip() for s in smiles.split(",") if s.strip()]
    
    if not smiles_list:
        return "Please enter at least one SMILES string"

    figure, summary = predict(smiles_list)
    return figure, summary


# ================================
# GRADIO UI
# ================================
demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(
        label="Enter SMILES (comma-separated)",
        placeholder="e.g. CCO, CCN, C1=CC=CC=C1"
    ),
    outputs=[gr.Plot(label="GNN prediction"), gr.Markdown(label="summary")],
    title="Molecular Solubility Predictor",
    description="Enter one or more SMILES strings separated by commas."
)


# ================================
# MOUNT GRADIO ON FASTAPI
# ================================
app = gr.mount_gradio_app(app, demo, path="/ui")