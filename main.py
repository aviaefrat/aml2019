from pathlib import Path
import os

from sklearn.model_selection import train_test_split
import pandas as pd

from constants import (DATA_DIR, ACTIONS_LIST,
                       ABC_LOWER, ACCENTS_LOWER, BASIC_PUNCTUATION, OTHER_SYMBOLS)
from data_loader import load_data
from preprocess import preprocess
from feature_extraction import NgramExtractor


def create_initial_data(dest_dir=DATA_DIR, seed=0):
    os.makedirs(dest_dir, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=0.2)

    X_train['tweet_text'].to_csv(os.path.join(dest_dir, 'raw_X_train.csv'), header=False)
    X_test['tweet_text'].to_csv(os.path.join(dest_dir, 'raw_X_test.csv'), header=False)
    y_train.to_csv(os.path.join(dest_dir, 'y_train.csv'), header=False)
    y_test.to_csv(os.path.join(dest_dir, 'y_test.csv'), header=False)


def create_preprocessed_data(data_dir=DATA_DIR, actions_list=ACTIONS_LIST, dest_dir=DATA_DIR):

    def save_preprocessed_data(train, test, dirpath):
        # remove tweets that end out as empty strings
        train = train[train != '']
        test = test[test != '']

        train.to_csv(os.path.join(dirpath, 'pre_X_train.csv'))
        test.to_csv(os.path.join(dirpath, 'pre_X_test.csv'))

    X_train = pd.read_csv(os.path.join(data_dir, 'raw_X_train.csv'), index_col=0, header=None, squeeze=True)
    X_test = pd.read_csv(os.path.join(data_dir, 'raw_X_test.csv'), index_col=0, header=None, squeeze=True)

    for actions in actions_list:
        actions_dirname = '-'.join(actions)
        actions_dirpath = os.path.join(dest_dir, actions_dirname)
        os.makedirs(actions_dirpath, exist_ok=True)

        preprocessed_X_train = preprocess(X_train, actions)
        preprocessed_X_test = preprocess(X_test, actions)
        save_preprocessed_data(preprocessed_X_train, preprocessed_X_test, actions_dirpath)

    # create a directory for the data without any pre processing
    no_preprocessing_dirpath = os.path.join(dest_dir, 'no_preprocessing')
    os.makedirs(no_preprocessing_dirpath, exist_ok=True)
    save_preprocessed_data(X_train, X_test, no_preprocessing_dirpath)


def create_featured_data(pre_processed_root=DATA_DIR, data_dir=DATA_DIR):
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0, header=None, squeeze=True)
    p = Path(pre_processed_root)
    ne = NgramExtractor()

    X_train = []
    X_test = []
    _1gram_chars = ABC_LOWER+ACCENTS_LOWER+OTHER_SYMBOLS
    _ngram_chars = ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION

    for dir_ in [d for d in p.iterdir() if d.is_dir()]:
        pre_X_train = pd.read_csv(os.path.join(dir_, 'pre_X_train.csv'), index_col=0, header=None, squeeze=True)
        pre_X_test = pd.read_csv(os.path.join(dir_, 'pre_X_test.csv'), index_col=0, header=None, squeeze=True)

        X_train.append(ne.fit_transform(pre_X_train, y_train, n=1, chars=_1gram_chars))
        X_test.append(ne.transform(pre_X_test))
        for n in range(2, 5+1):
            X_train.append(ne.fit_transform(pre_X_train, y_train, n=n, chars=_ngram_chars))
            X_test.append(ne.transform(pre_X_test))

        X_train = pd.concat(X_train, axis=1)
        X_train.to_csv(os.path.join(dir_, 'X_train.csv'))
        X_test = pd.concat(X_test, axis=1)
        X_test.to_csv(os.path.join(dir_, 'X_test.csv'))


# create_initial_data()
# create_preprocessed_data()
ne = create_featured_data()