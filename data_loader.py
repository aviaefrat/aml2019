import os

import pandas as pd


DATA_DIR = os.path.join(os.getcwd(), 'dataset')
LANGUAGES = ('en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl')


def _load_language_data(lang, data_dir=DATA_DIR):
    filepath = f'{os.path.join(data_dir, lang)}.csv'
    df = pd.read_csv(filepath, usecols=['tweet_text'])
    return df


def load_data(langs=LANGUAGES, data_dir=DATA_DIR):
    languages_df = []
    for lang in langs:
        language_df = _load_language_data(lang, data_dir)
        languages_df.append(language_df)
        # add label
        language_df['lang'] = lang
    data = pd.concat(languages_df, axis=0)
    return data
