import pandas as pd
from skimage.morphology import disk

def get_car_numbers_dataset():
    data = pd.read_csv('data/ocr_car_numbers_rotulado.txt', delim_whitespace=True, header=None)
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    return X, y
