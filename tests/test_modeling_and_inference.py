import os
import sys
import math
import matplotlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Model.model import dmpnn
from src.Model.trainer import start_training
from src.Model.tuining import tune_model
from src.serving.inference import predict_solubility
from src.serving.plot import build_plot
from src.serving.llm_ import llm

def start_model_testing(
    train_loader,
    test_loader,
    valid_loader,
    target_scaler,
    mol_feat_size,
    extra_feat_scaler
):
    best_value, params = tune_model(
        train_loader,
        valid_loader,
        target_scaler,
        mol_feat_size=mol_feat_size,
        enable_mlflow=False,
        epoches=10,
        n_trials=10
    ).start_tuining()

    assert isinstance(best_value, (int, float))
    assert math.isfinite(best_value)

    assert isinstance(params, dict)
    assert len(params) > 0

    model = dmpnn(
        scale=target_scaler,
        mol_feat_size=mol_feat_size,
        **params
    )

    assert model is not None

    trainer = start_training(
        model,
        train_loader,
        valid_loader
    )

    assert trainer is not None

    results = trainer.test(
        model=model,
        dataloaders=test_loader
    )

    metrics = results[0]

    assert isinstance(metrics, dict)
    assert len(metrics) > 0

    for metric_name, metric_value in metrics.items():

        assert isinstance(metric_name, str)

        assert isinstance(metric_value, (int, float))

        assert math.isfinite(metric_value)

    print("Testing successful")
    
    test_inputs = ["CCO", "invalid_smiles", "CCN"]
    solubility, valid_smiles = predict_solubility(test_inputs, testing=True, test_model=model, test_scaler=extra_feat_scaler)
    
    assert len(solubility) == 2
    for i, j in zip(solubility, valid_smiles):
        assert isinstance(i, (int, float))
        assert isinstance(j, str)
    
    fig = build_plot(valid_smiles, solubility)
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(ax.lines) == len(solubility)

    llm_content = llm(valid_smiles, solubility)
    assert isinstance(llm_content, str)
    assert len(llm_content)>0
    return None