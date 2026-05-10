import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Data.preprocessing import preprocess_data


def start_preprocess_testing(train, test, valid):

    preprocessor = preprocess_data(train, test, valid)

    # =========================
    # Feature Extraction
    # =========================
    train_mols, train_y, train_mol_feats = preprocessor.extract_features(train)
    test_mols, test_y, test_mol_feats = preprocessor.extract_features(test)
    valid_mols, valid_y, valid_mol_feats = preprocessor.extract_features(valid)

    # Assertions after extraction
    assert len(train_mols) > 0, "Train molecules are empty"
    assert len(test_mols) > 0, "Test molecules are empty"
    assert len(valid_mols) > 0, "Valid molecules are empty"

    assert len(train_mols) == len(train_y) == len(train_mol_feats)
    assert len(test_mols) == len(test_y) == len(test_mol_feats)
    assert len(valid_mols) == len(valid_y) == len(valid_mol_feats)

    assert isinstance(train_mol_feats, np.ndarray)
    assert isinstance(test_mol_feats, np.ndarray)
    assert isinstance(valid_mol_feats, np.ndarray)

    # =========================
    # Feature Normalization
    # =========================
    train_mol_feats, test_mol_feats, valid_mol_feats = (
        preprocessor.normalize_extra_mol_feats(
            train_mol_feats,
            test_mol_feats,
            valid_mol_feats
        )
    )

    # Assertions after normalization
    assert train_mol_feats.shape[1] == test_mol_feats.shape[1]
    assert train_mol_feats.shape[1] == valid_mol_feats.shape[1]

    assert preprocessor.mol_feat_size == train_mol_feats.shape[1]

    # =========================
    # Datapoint Creation
    # =========================
    train_dp = preprocessor.create_datapoint(
        train_mols,
        train_y,
        train_mol_feats
    )

    test_dp = preprocessor.create_datapoint(
        test_mols,
        test_y,
        test_mol_feats
    )

    valid_dp = preprocessor.create_datapoint(
        valid_mols,
        valid_y,
        valid_mol_feats
    )

    # Assertions after datapoint creation
    assert len(train_dp) == len(train_mols)
    assert len(test_dp) == len(test_mols)
    assert len(valid_dp) == len(valid_mols)

    # =========================
    # Dataset Featurization
    # =========================
    train_ds, test_ds, valid_ds = preprocessor.initiate_featurizer(
        train_dp,
        test_dp,
        valid_dp
    )

    assert len(train_ds) == len(train_dp)
    assert len(test_ds) == len(test_dp)
    assert len(valid_ds) == len(valid_dp)

    # =========================
    # Target Normalization
    # =========================
    train_ds, valid_ds = preprocessor.normalizing_target(
        train_ds,
        valid_ds
    )

    assert preprocessor.target_scaler is not None

    # =========================
    # Loader Building
    # =========================
    train_loader = preprocessor.build_loader(train_ds)
    test_loader = preprocessor.build_loader(test_ds, shuffle=False)
    valid_loader = preprocessor.build_loader(valid_ds)

    # Assertions for loaders
    assert train_loader is not None
    assert test_loader is not None
    assert valid_loader is not None

    return (
        train_loader,
        test_loader,
        valid_loader,
        preprocessor.target_scaler,
        preprocessor.mol_feat_size,
        preprocessor.extra_feat_scaler
    )