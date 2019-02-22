import os

from sklearn.model_selection import train_test_split
import numpy as np

from constants import DATA_DIR
from data_loader import load_data


def create_initial_data(data_dir=None, seed=0):
    data_dir = data_dir or os.path.join(os.getcwd(), DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=0.2)

    X_train.to_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test.to_csv(os.path.join(data_dir, 'X_test.csv'))
    np.save(os.path.join(data_dir, 'y_train'), y_train)
    np.save(os.path.join(data_dir, 'y_test'), y_test)


create_initial_data()
