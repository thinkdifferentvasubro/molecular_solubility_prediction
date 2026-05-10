from src.serving.inference import predict_solubility
from src.serving.llm_ import llm
from src.serving.plot import build_plot
import logging
import warnings


def predict(smiles):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    solubility, valid_smiles = predict_solubility(smiles)
    fig = build_plot(valid_smiles, solubility)
    final_prediction = llm(valid_smiles, solubility)
    return fig, final_prediction