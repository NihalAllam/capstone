import pandas as pd


def load_pima(path="../data/PIMA.csv"):
    df = pd.read_csv(path)
    return df
