import os

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from constants import DATA_DIR, ACTIONS_LIST
from data_loader import load_data
from preprocess import preprocess


def create_initial_data(dest_dir=DATA_DIR, seed=0):
    os.makedirs(dest_dir, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=0.2)

    X_train['tweet_text'].to_csv(os.path.join(dest_dir, 'X_train.csv'), header=False)
    X_test['tweet_text'].to_csv(os.path.join(dest_dir, 'X_test.csv'), header=False)
    np.save(os.path.join(dest_dir, 'y_train'), y_train)
    np.save(os.path.join(dest_dir, 'y_test'), y_test)


def create_preprocessed_data(data_dir=DATA_DIR, actions_list=ACTIONS_LIST, dest_dir=DATA_DIR):

    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), index_col=0, header=None, squeeze=True)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), index_col=0, header=None, squeeze=True)

    for actions in actions_list:
        actions_dirname = '-'.join(actions)
        actions_dirpath = os.path.join(dest_dir, actions_dirname)
        os.makedirs(actions_dirpath, exist_ok=True)

        preprocessed_X_train = preprocess(X_train, actions)
        preprocessed_X_test = preprocess(X_test, actions)

        preprocessed_X_train.to_csv(os.path.join(actions_dirpath, 'X_train.csv'))
        preprocessed_X_test.to_csv(os.path.join(actions_dirpath, 'X_test.csv'))


# create_initial_data()
# create_preprocessed_data()
