import lightgbm as lgb

from data_loader import load_data
from preprocess import (create_char_proportions_feature, create_unicode_block_proportions_feature,
                        extract_char_ngrams_feature, remove_retweets, remove_urls)


data = load_data()
X = data.drop(['lang'], axis=1)
y, _ = data['lang'].factorize()

X['tweet_text'] = remove_retweets(X['tweet_text'])
X['tweet_text'] = remove_urls(X['tweet_text'])
X = create_unicode_block_proportions_feature(X)
X = create_char_proportions_feature(X)
X = extract_char_ngrams_feature(X, chars=[chr(i) for i in range(32, 47)] + [chr(i) for i in range(91, 126)], n=2)
X = X.drop(['tweet_text'], axis=1)

train_data = lgb.Dataset(X, label=y)

num_round = 10
param = {'num_leaves': 31, 'num_trees': 150, 'objective': 'multiclass', 'num_class': 8,
         'metrics': 'multi_error'}

result = lgb.cv(param, train_data, num_round, nfold=3)
