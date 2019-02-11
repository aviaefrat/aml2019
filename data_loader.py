import os

import pandas as pd


DATA_DIR = os.path.join(os.getcwd(), 'dataset')


def load_language_data(lang, data_dir=DATA_DIR):
    filepath = f'{os.path.join(data_dir, lang)}.csv'
    df = pd.read_csv(filepath, usecols=['tweet_text'])
    return df



