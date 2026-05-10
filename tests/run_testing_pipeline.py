import os
import sys
from test_preprocessing import start_preprocess_testing
from test_modeling_and_inference import start_model_testing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.Data.load import load_data

train, test, valid = load_data()
train = train.iloc[:30, :]
test = test.iloc[:7, :]
valid = valid.iloc[:7, :]
train_loader, test_loader, valid_loader, target_scaler, mol_feat_size, extra_feat_scaler = start_preprocess_testing(train, test, valid)
start_model_testing(
        train_loader,
        test_loader,
        valid_loader,
        target_scaler,
        mol_feat_size,
        extra_feat_scaler
)
