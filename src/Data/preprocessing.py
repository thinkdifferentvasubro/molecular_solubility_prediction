import random
import os
import pickle
import numpy as np
from chemprop import data, featurizers, utils
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from sklearn.compose import ColumnTransformer


class preprocess_data:

    def __init__(self, train, test, valid, smiles="smiles",target="measured log solubility in mols per litre"):
        self.train = train
        self.test = test
        self.valid = valid
        self.target = target.strip()
        self.smiles = smiles.strip()
        self.target_scaler = None
        self.mol_feat_size = None
        self.extra_feat_scaler = None

    def run_pipeline(self):
        self.set_seed()

        #extracting features
        train_mols, train_y, train_mol_feats = self.extract_features(self.train)
        test_mols, test_y, test_mol_feats = self.extract_features(self.test)
        valid_mols, valid_y, valid_mol_feats = self.extract_features(self.valid)

        #normalizing extra features
        train_mol_feats, test_mol_feats, valid_mol_feats = self.normalize_extra_mol_feats(train_mol_feats, test_mol_feats, valid_mol_feats)
        
        #creating datapoint
        train_dp = self.create_datapoint(train_mols, train_y, train_mol_feats)
        test_dp = self.create_datapoint(test_mols, test_y, test_mol_feats)
        valid_dp = self.create_datapoint(valid_mols, valid_y, valid_mol_feats)

        #initilizing features
        train_ds, test_ds, valid_ds = self.initiate_featurizer(train_dp, test_dp, valid_dp)

        #normalizing target
        train_ds, valid_ds = self.normalizing_target(train_ds, valid_ds)

        #building loaders
        train_loader = self.build_loader(train_ds)
        test_loader = self.build_loader(test_ds, shuffle=False)
        valid_loader = self.build_loader(valid_ds)

        return train_loader, test_loader, valid_loader, self.target_scaler, self.mol_feat_size, self.extra_feat_scaler

    def build_loader(self, dataset, shuffle=True):
        return data.build_dataloader(dataset, batch_size=32, shuffle=shuffle)
    
    def normalize_extra_mol_feats(self, train_extra_features, test_extra_features, valid_extra_features):
        scaler = StandardScaler()
        feats_to_normalize = []
        for feat_inx in range(train_extra_features.shape[1]):
            unique_feat_values = np.unique(train_extra_features[:, feat_inx])
            if len(unique_feat_values) > 19:
                feats_to_normalize.append(feat_inx)
        if len(feats_to_normalize) == 0:
            print("No features to normalize")
            self.mol_feat_size = train_extra_features.shape[1]
            return train_extra_features, test_extra_features, valid_extra_features
        transformer = ColumnTransformer(
            transformers=[
                ("num_scaler", scaler, feats_to_normalize)
                ],
                remainder='passthrough'
                )
        transformer.fit(train_extra_features)
        train_extra_features = transformer.transform(train_extra_features)
        test_extra_features = transformer.transform(test_extra_features)
        valid_extra_features = transformer.transform(valid_extra_features)
        self.mol_feat_size = train_extra_features.shape[1]
        self.extra_feat_scaler = transformer
        return train_extra_features, test_extra_features, valid_extra_features

    def normalizing_target(self, train_ds, val_ds):
        scaler = train_ds.normalize_targets()
        self.target_scaler = scaler
        val_ds.normalize_targets(scaler)

        return train_ds, val_ds

    def initiate_featurizer(self, train_dp, test_dp, valid_dp):
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_ds = data.MoleculeDataset(train_dp, featurizer=featurizer)
        test_ds = data.MoleculeDataset(test_dp, featurizer=featurizer)
        valid_ds = data.MoleculeDataset(valid_dp, featurizer=featurizer)
        return train_ds, test_ds, valid_ds

    def create_datapoint(self, mols, y, mol_feats):
        return [
            data.MoleculeDatapoint(mol, [y_], x_d=X_d)
            for mol, y_, X_d in zip(mols, y, mol_feats)
        ]

    def extract_features(self, dataset):
        dataset = self.filter_smiles(dataset)

        y = dataset[self.target].values
        smiles = dataset[self.smiles].values

        mols, ys, fs = [], [], []

        for smi, t in zip(smiles, y):
            try:
                mol = utils.make_mol(smi, keep_h=True, add_h=True)
                mol_feat = self.build_extra_feats_from_smiles(smi)
                mols.append(mol)
                ys.append(t)
                fs.append(mol_feat)
            except:
                continue
        return mols, np.array(ys), np.array(fs)

    def filter_smiles(self, dataset):
        dataset = dataset.dropna()
        dataset = dataset[
            dataset[self.smiles].apply(lambda x: Chem.MolFromSmiles(x) is not None)
        ].reset_index(drop=True)
        if len(dataset)==0:
            raise Exception("Your error message here")
        return dataset
    
    def build_extra_feats_from_smiles(self, smile):
        mol_ = Chem.MolFromSmiles(smile)
        molecular_weight = Descriptors.MolWt(mol_)
        num_h_donors = rdMolDescriptors.CalcNumHBD(mol_)
        num_rings = rdMolDescriptors.CalcNumRings(mol_)
        num_rotatable_rings = rdMolDescriptors.CalcNumRotatableBonds(mol_)
        tpsa = rdMolDescriptors.CalcTPSA(mol_)
        degrees = [atom.GetDegree() for atom in mol_.GetAtoms()]
        min_degree = min(degrees) if degrees else 0

        mol_feat = [
            molecular_weight, 
            num_h_donors, 
            num_rings, 
            num_rotatable_rings,
            tpsa,
            min_degree 
            ]
        return mol_feat

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)