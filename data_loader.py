import os

import pandas as pd

from constants import LANGUAGES


DATA_DIR = os.path.join(os.getcwd(), 'dataset')


def _load_language_data(lang, data_dir=DATA_DIR):
    filepath = f'{os.path.join(data_dir, lang)}.csv'
    df = pd.read_csv(filepath, usecols=['tweet_text'])
    return df


def load_data(langs=LANGUAGES, data_dir=DATA_DIR, lang_sample_size=-1):
    languages_df = []
    for lang in langs:
        language_df = _load_language_data(lang, data_dir)

        # subsample language data
        if lang_sample_size > 0:
            language_df = language_df.sample(lang_sample_size)

        # add label
        language_df['language_id'] = lang

        languages_df.append(language_df)

    data = pd.concat(languages_df, axis=0, ignore_index=True)
    return data
