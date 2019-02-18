import lightgbm as lgb
import pandas as pd

from constants import ABC, ABC_LOWER, ACCENTS, ACCENTS_LOWER, BASIC_PUNCTUATION, OTHER_SYMBOLS
from data_loader import load_data
from preprocess import (create_unicode_block_proportions_feature,
                        extract_char_ngrams_feature, remove_retweets, remove_urls, remove_handles,
                        reduce_lengthening, extract_ngrams_feature)


data = load_data()
X = data
y, _ = data['lang'].factorize()

X['tweet_text'] = remove_retweets(X['tweet_text'])
X['tweet_text'] = remove_handles(X['tweet_text'])
X['tweet_text'] = remove_urls(X['tweet_text'])
X['tweet_text'] = reduce_lengthening(X['tweet_text'])

X = create_unicode_block_proportions_feature(X)
X = extract_char_ngrams_feature(X, chars=ABC + ACCENTS, n=1, ignore_case=False, ratio_of_total=False)
X = extract_char_ngrams_feature(X, chars=OTHER_SYMBOLS, n=1, ignore_case=False, ratio_of_total=False)
X = extract_char_ngrams_feature(X, chars=ABC_LOWER + ACCENTS_LOWER + BASIC_PUNCTUATION, n=2, ratio_of_total=False)
X = pd.concat([X, extract_ngrams_feature(X, 3, ABC_LOWER+ACCENTS_LOWER+' ', 500)], axis=1)
X = X.drop(['tweet_text', 'lang'], axis=1)

train_data = lgb.Dataset(X, label=y)

num_round = 10
param = {'num_leaves': 31, 'num_trees': 150, 'objective': 'multiclass', 'num_class': 8,
         'metrics': 'multi_error'}

result = lgb.cv(param, train_data, num_round, nfold=3)
