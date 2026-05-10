import optuna
import mlflow

from src.Model.model import dmpnn
from src.Model.trainer import start_training


class tune_model:
    def __init__(
        self,
        train_loader,
        val_loader,
        scaler=None,
        direction="minimize",
        n_trials=30,
        mol_feat_size=None,
        epoches=None,
        enable_mlflow=True
    ):
        self.scaler = scaler
        self.direction = direction
        self.n_trials = n_trials
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mol_feat_size = mol_feat_size
        self.epoches = epoches
        self.enable_mlflow = enable_mlflow

        if self.enable_mlflow:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("optuna_search")

    def start_tuining(self):

        if self.enable_mlflow:

            with mlflow.start_run(run_name="best_params"):

                study = optuna.create_study(direction=self.direction)
                study.optimize(self.objective, n_trials=self.n_trials)

                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_value", study.best_value)

        else:

            study = optuna.create_study(direction=self.direction)
            study.optimize(self.objective, n_trials=self.n_trials)

        return study.best_trial.value, study.best_trial.params

    def objective(self, trial):

        depth = trial.suggest_int("depth", 2, 8)
        dropout = trial.suggest_float("dropout", 0.05, 0.5)
        dropout_fnn = trial.suggest_float("dropout_fnn", 0.05, 0.5)
        d_h = trial.suggest_int("d_h", 100, 300)
        hidden_dim = trial.suggest_int("hidden_dim", 100, 300)
        n_layers = trial.suggest_int("n_layers", 1, 7)

        model = dmpnn(
            self.scaler,
            depth,
            dropout,
            dropout_fnn,
            d_h,
            hidden_dim,
            n_layers,
            mol_feat_size=self.mol_feat_size
        )

        trainer = start_training(
            model,
            self.train_loader,
            self.val_loader
        )

        val_mae = trainer.callback_metrics["val/mae"].item()
        val_rmse = trainer.callback_metrics["val/rmse"].item()

        if self.enable_mlflow:

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True
            ):

                mlflow.log_params({
                    "trial_number": trial.number,
                    "depth": depth,
                    "dropout": dropout,
                    "dropout_fnn": dropout_fnn,
                    "d_h": d_h,
                    "hidden_dim": hidden_dim,
                    "n_layers": n_layers,
                })

                mlflow.log_metric("val_mae", val_mae)
                mlflow.log_metric("val_rmse", val_rmse)

        return val_mae