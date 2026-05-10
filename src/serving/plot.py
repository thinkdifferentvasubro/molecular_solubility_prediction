import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.Data.load import load_data


def build_plot(smiles, solubility_predicted):
    train, _, _ = load_data()
    measured_solubility = train["measured log solubility in mols per litre"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        measured_solubility,
        bins=30,
        color="#4A90E2",
        edgecolor="white",
        alpha=0.5
    )

    colors = ["#0B3D91", "#1E90FF", "#003F7F", "#1565C0"]

    for i, pred in enumerate(solubility_predicted):
        ax.axvline(
            pred,
            color=colors[i % len(colors)],
            linestyle="dashed",
            linewidth=2,
            label=f"{smiles[i]} :- {solubility_predicted[i]}"
        )

    ax.set_xlabel("logS (solubility)", fontsize=12)
    ax.set_title("Solubility Distribution", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.2)

    return fig  # 🔥 THIS is the key change