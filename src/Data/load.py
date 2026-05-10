import pandas as pd

def load_data(
        train_path="hf://datasets/HR-machine/ESol/train_data.csv",
        test_path = "hf://datasets/HR-machine/ESol/test_data.csv", 
        validation_path = "hf://datasets/HR-machine/ESol/valid_data.csv"
        ):
    """
    Load train, test and validation datasets from Hugging Face CSV files.
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    valid = pd.read_csv(validation_path)

    return train, test, valid