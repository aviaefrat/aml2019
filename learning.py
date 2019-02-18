import lightgbm as lgb
import pandas as pd

from constants import ABC, ABC_LOWER, ACCENTS, ACCENTS_LOWER, BASIC_PUNCTUATION, OTHER_SYMBOLS
from data_loader import load_data
from preprocess import (create_unicode_block_proportions_feature,  remove_retweets, remove_urls,
                        remove_handles, reduce_lengthening, extract_ngrams_feature)


data = load_data()
X = data
y, _ = data['language_id'].factorize()

X['tweet_text'] = remove_retweets(X['tweet_text'])
X['tweet_text'] = remove_handles(X['tweet_text'])
X['tweet_text'] = remove_urls(X['tweet_text'])
X['tweet_text'] = reduce_lengthening(X['tweet_text'])

X = create_unicode_block_proportions_feature(X)
X = pd.concat([X, extract_ngrams_feature(X, 1, ABC_LOWER+ACCENTS_LOWER, k='all')], axis=1)
X = pd.concat([X, extract_ngrams_feature(X, 1, OTHER_SYMBOLS, k='all')], axis=1)
X = pd.concat([X, extract_ngrams_feature(X, 2, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 1000)], axis=1)
X = pd.concat([X, extract_ngrams_feature(X, 3, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 500)], axis=1)
X = pd.concat([X, extract_ngrams_feature(X, 4, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 500)], axis=1)
X = pd.concat([X, extract_ngrams_feature(X, 5, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 500)], axis=1)
X = X.drop(['tweet_text', 'language_id'], axis=1)

train_data = lgb.Dataset(X, label=y)

num_round = 10
param = {'num_leaves': 31, 'num_trees': 150, 'objective': 'multiclass', 'num_class': 8,
         'metrics': 'multi_error'}

result = lgb.cv(param, train_data, num_round, nfold=3)
