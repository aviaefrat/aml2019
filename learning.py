import lightgbm as lgb

from constants import abc, abc_lower, accents, accents_lower, basic_punctuation, other_symbols
from data_loader import load_data
from preprocess import (create_unicode_block_proportions_feature,
                        extract_char_ngrams_feature, remove_retweets, remove_urls,
                        extract_n_hashtags_feature, remove_handles, reduce_lengthening)


data = load_data()
X = data.drop(['lang'], axis=1)
y, _ = data['lang'].factorize()

X['tweet_text'] = remove_retweets(X['tweet_text'])
X['tweet_text'] = remove_handles(X['tweet_text'])
X['tweet_text'] = remove_urls(X['tweet_text'])
X['tweet_text'] = reduce_lengthening(X['tweet_text'])

X = create_unicode_block_proportions_feature(X)
X = extract_char_ngrams_feature(X, chars=abc+accents, n=1, ignore_case=False, ratio_of_total=False)
X = extract_char_ngrams_feature(X, chars=other_symbols, n=1, ignore_case=False, ratio_of_total=False)
X = extract_char_ngrams_feature(X, chars=abc_lower+accents_lower+basic_punctuation, n=2, ratio_of_total=False)
X = X.drop(['tweet_text'], axis=1)

train_data = lgb.Dataset(X, label=y)

num_round = 10
param = {'num_leaves': 31, 'num_trees': 150, 'objective': 'multiclass', 'num_class': 8,
         'metrics': 'multi_error'}

result = lgb.cv(param, train_data, num_round, nfold=3)
