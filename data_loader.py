import os

import numpy as np
import pandas as pd

from constants import LANGUAGES


DATA_DIR = os.path.join(os.getcwd(), 'dataset')


def _load_language_data(lang, data_dir=DATA_DIR):
    filepath = f'{os.path.join(data_dir, lang)}.csv'
    df = pd.read_csv(filepath, usecols=['tweet_text'])
    return df


def load_data(langs=LANGUAGES, data_dir=DATA_DIR, lang_sample_size=-1):
    X = []
    y = []
    for lang in langs:
        language_df = _load_language_data(lang, data_dir)

        # subsample language data
        if lang_sample_size > 0:
            language_df = language_df.sample(lang_sample_size)

        X.append(language_df)
        y.append(np.full(len(language_df), lang))

    X = pd.concat(X, axis=0, ignore_index=True)
    y = pd.Series(np.concatenate(y), index=X.index)
    return X, y
