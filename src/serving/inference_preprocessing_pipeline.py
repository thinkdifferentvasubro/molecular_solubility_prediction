import random
import numpy as np
from chemprop import data, featurizers, utils
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


class preprocess_inference_data:

    def __init__(self, extra_feats_scaler=None):
        self.extra_feats_scaler = extra_feats_scaler

    def run_pipeline(self, smiles):
        smiles = [smile.strip() for smile in smiles]
        self.set_seed()

        #filtering smiles
        smiles = self.filter_smiles(smiles)

        #extract features
        mols, extra_features, valid_smiles = self.extract_features(smiles)

        #normalizing_extra_features
        extra_features = self.normalize_extra_mol_feats(extra_features)

        #creating datapoint
        datapoint = self.create_datapoint(mols, extra_features)

        #applying featurizer
        featurized_data = self.initiate_featurizer(datapoint)

        #build_loader
        loader = self.build_loader(featurized_data)

        return loader, valid_smiles

    def build_loader(self, data_):
        return data.build_dataloader(data_, shuffle=False)
    
    def normalize_extra_mol_feats(self, extra_features):
        extra_features = self.extra_feats_scaler.transform(extra_features)
        return extra_features
    
    def initiate_featurizer(self, datapoint):
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        featurized_data = data.MoleculeDataset(datapoint, featurizer=featurizer)
        return featurized_data

    def create_datapoint(self, mols, mol_feats):
        return [
            data.MoleculeDatapoint(mol, x_d=X_d)
            for mol, X_d in zip(mols, mol_feats)
        ]

    def extract_features(self, smiles):
        mols, fs, valid_smiles= [], [], []

        for smi in smiles:
            try:
                mol = utils.make_mol(smi, keep_h=False, add_h=False)
                valid_smiles.append(smi)
                mol_feat = self.build_extra_feats_from_smiles(smi)
                mols.append(mol)
                fs.append(mol_feat)
            except:
                continue
        return mols, np.array(fs), valid_smiles

    def filter_smiles(self, smiles):
        filtered_smiles = []
        for smile in smiles:
            smile = smile.strip()
            if Chem.MolFromSmiles(smile) is not None:
                filtered_smiles.append(smile)
        if len(filtered_smiles)==0:
            raise Exception("at least pass one valid smile")
        return filtered_smiles
    
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