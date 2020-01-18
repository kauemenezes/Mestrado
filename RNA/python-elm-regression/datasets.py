import numpy as np
import random
import pandas as pd
from classifier import Classifier


def get_artificial_one():
    X = np.linspace(-5, 5, 500).reshape((-1, 1))
    Y = []
    for i in range(len(X)):
        Y.append(3 * np.sin(X[i]) + 1 + random.uniform(-1, 1))

    dataset = np.concatenate([X.reshape(-1, 1), np.array(Y).reshape(-1, 1)], axis=1)
    return dataset


def get_car_fuel_consumption_dataset():
    names = ['distance', 'speed', 'temp_inside', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun', 'consume']
    # loading data
    dataset = pd.read_csv('measurements.csv', names=names).values
    dataset = np.array(dataset, dtype=float)
    dataset = Classifier.normalize_dataset(dataset)

    return dataset


def get_abalone_dataset():
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    # loading data
    dataset = pd.read_csv('abalone.csv', names=names).values
    dataset = Classifier.normalize_dataset(dataset)

    return dataset


def get_eletric_motor_temperature_dataset():
    names = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding', 'profile_id', 'pm']
    # loading data
    dataset = pd.read_csv('pmsm_temperature_data.csv', names=names).values
    dataset = Classifier.normalize_dataset(dataset)

    return dataset
