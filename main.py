import os

from sklearn.model_selection import train_test_split
import pandas as pd

from constants import DATA_DIR, ACTIONS_LIST, ABC_LOWER, ACCENTS_LOWER
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


def create_featured_data():
    ne = NgramExtractor()
    X_train = pd.read_csv(os.path.join(os.getcwd(), 'data', 'rt-handle-letter_repeat-url', 'pre_X_train.csv'), index_col=0, header=None, squeeze=True)
    y_train = pd.read_csv(os.path.join(os.getcwd(), 'data', 'y_train.csv'), index_col=0, header=None, squeeze=True)
    ne.fit(X_train, y_train, 3, chars=ABC_LOWER+ACCENTS_LOWER)
    return ne

create_initial_data()
create_preprocessed_data()
# ne = create_featured_data()