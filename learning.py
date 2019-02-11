import lightgbm as lgb

from data_loader import load_data
from preprocess import create_char_proportions_features, create_unicode_block_proportions_feature


data = load_data()
X = data.drop(['lang'], axis=1)
y, _ = data['lang'].factorize()

X = create_unicode_block_proportions_feature(X)
X = create_char_proportions_features(X)
X = X.drop(['tweet_text'], axis=1)

train_data = lgb.Dataset(X, label=y)

num_round = 10
param = {'num_leaves': 31, 'num_trees': 150, 'objective': 'multiclass', 'num_class': 8,
         'metrics': 'multi_error'}

result = lgb.cv(param, train_data, num_round, nfold=3)
