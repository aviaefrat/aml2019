from collections import defaultdict
from pathlib import Path
import os
import pickle

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd

from constants import (DATA_DIR, MODELS_DIR, RESULTS_DIR,
                       LANGUAGES, ACTIONS_LIST, FEATURE_TYPES,
                       ABC, ACCENTS, ABC_LOWER, ACCENTS_LOWER, BASIC_PUNCTUATION, OTHER_SYMBOLS,
                       HPARAM_GRID, CONSTANT_HPARAMS)
from data_loader import load_data
from preprocess import preprocess
from feature_extraction import NgramExtractor, VocabExtractor, get_features


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
        # remove tweets that end out as containing no letters
        train = train[train.str.contains(f"[{ABC+ACCENTS}]")]
        test = test[test.str.contains(f"[{ABC+ACCENTS}]")]

        train.to_csv(os.path.join(dirpath, 'pre_X_train.csv'), header=False)
        test.to_csv(os.path.join(dirpath, 'pre_X_test.csv'), header=False)

    X_train = pd.read_csv(os.path.join(data_dir, 'raw_X_train.csv'), index_col=0, header=None, squeeze=True, quotechar='"')
    X_test = pd.read_csv(os.path.join(data_dir, 'raw_X_test.csv'), index_col=0, header=None, squeeze=True, quotechar='"')

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
    _1gram_chars = ABC_LOWER+ACCENTS_LOWER+OTHER_SYMBOLS
    _ngram_chars = ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION

    ve = VocabExtractor()

    for dir_ in [d for d in p.iterdir() if d.is_dir()]:
        pre_X_train = pd.read_csv(os.path.join(dir_, 'pre_X_train.csv'), index_col=0, header=None, squeeze=True, quotechar='"')
        pre_X_test = pd.read_csv(os.path.join(dir_, 'pre_X_test.csv'), index_col=0, header=None, squeeze=True, quotechar='"')
        X_train = []
        X_test = []

        # extract char ngrams
        X_train.append(ne.fit_transform(pre_X_train, y_train, n=1, chars=_1gram_chars))
        X_test.append(ne.transform(pre_X_test))
        for n in range(2, 5+1):
            X_train.append(ne.fit_transform(pre_X_train, y_train, n=n, chars=_ngram_chars))
            X_test.append(ne.transform(pre_X_test))

        # extract words and hashtags
        X_train.append(ve.fit_transform(pre_X_train, y_train))
        X_test.append(ve.transform(pre_X_test))

        # save featured data
        X_train = pd.concat(X_train, axis=1)
        X_train.to_csv(os.path.join(dir_, 'X_train.csv'))
        X_test = pd.concat(X_test, axis=1)
        X_test.to_csv(os.path.join(dir_, 'X_test.csv'))


# create_initial_data()
# create_preprocessed_data()
# create_featured_data()


def tune_hyperparams(train_data, param_grid=HPARAM_GRID):
    cv_results = {}
    for params in param_grid:
        params.update(CONSTANT_HPARAMS)
        print(f"performing CV with params:\n{params}")
        cv_result = lgb.cv(params, train_data, nfold=5)

        # get the optimal number of rounds from early stopping
        metric = f"{params['metric']}-mean"
        num_rounds = len(cv_result[metric])
        params['num_iterations'] = num_rounds
        del params['early_stopping_round']

        # save the score of these params
        best_score = cv_result[metric][-1]
        cv_results[best_score] = params

    # return the best score and its params
    best_score = min(cv_results)
    print(f"CV results: {cv_results}")
    print(f"best CV score: {best_score}.\nbest CV params: {cv_results[best_score]}")
    return best_score, cv_results[best_score]


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


def train_and_test(data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    p = Path(data_dir)
    y_train_ = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col=0, header=None, squeeze=True)
    y_train_ = y_train_.map(LANGUAGES)
    y_test_ = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col=0, header=None, squeeze=True)
    y_test_ = y_test_.map(LANGUAGES)
    results = _recursive_defaultdict()

    for dir_ in [d for d in p.iterdir() if d.is_dir()]:

        # load the relevant preprocessed data
        print(f"loading X_train for {dir_.name}")
        X_train = pd.read_csv(os.path.join(dir_, 'X_train.csv'), index_col=0)
        print(f"indexing y_train for {dir_.name}")
        y_train = y_train_.loc[X_train.index]
        results[dir_.name]['y_train'] = y_train
        print(f"loading X_test for {dir_.name}")
        X_test = pd.read_csv(os.path.join(dir_, 'X_test.csv'), index_col=0)
        print(f"indexing y_test for {dir_.name}")
        y_test = y_test_.loc[X_test.index]
        results[dir_.name]['y_test'] = y_test

        for feature_type in FEATURE_TYPES:
            # select features to use in training
            features = get_features(X_train, type_=feature_type)
            print(f"using features `{feature_type}` with preprocessing `{dir_.name}`")
            train_data = lgb.Dataset(features, label=y_train)

            # tune hyper parameters and save them
            best_score, best_hyperparams = tune_hyperparams(train_data)
            results[dir_.name][feature_type]['cv_error'] = best_score
            results[dir_.name][feature_type]['params'] = best_hyperparams

            # train
            print(f"training using features `{feature_type}` with preprocessing `{dir_.name}`")
            print(f"training params:\n {best_hyperparams}")
            booster = lgb.train(best_hyperparams, train_data)

            # calculate training metrics
            train_predictions = booster.predict(X_train).argmax(axis=1)
            train_predictions = pd.Series(train_predictions, index=X_train.index)
            results[dir_.name][feature_type]['train_pred'] = train_predictions
            # todo delete the following line later
            results[dir_.name][feature_type]['train_error'] = np.mean(train_predictions != y_train)
            print(f"training error is {results[dir_.name][feature_type]['train_error']}")

            # calculate test metrics
            test_predictions = booster.predict(X_test).argmax(axis=1)
            test_predictions = pd.Series(test_predictions, index=X_test.index)
            results[dir_.name][feature_type]['test_pred'] = test_predictions
            # todo delete the following line later
            results[dir_.name][feature_type]['test_error'] = np.mean(test_predictions != y_test)
            print(f"test error is {results[dir_.name][feature_type]['test_error']}")
            # save the model
            models_dir = os.path.join(dir_, MODELS_DIR)
            os.makedirs(models_dir, exist_ok=True)
            booster.save_model(os.path.join(models_dir, f"{feature_type}_model.txt"))

    # save the results
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle)

    return results


results = train_and_test()