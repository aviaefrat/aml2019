import lightgbm as lgb

from constants import ABC_LOWER, ACCENTS_LOWER, BASIC_PUNCTUATION, OTHER_SYMBOLS
from data_loader import load_data
from preprocess import PreProcessor
from feature_extraction import FeatureExtractor

data = load_data(lang_sample_size=100)
X = data['tweet_text']
y = data['language_id']

# pp = PreProcessor(X)
# X = pp.preprocess()
# loaded = pp.load()
#
# X = remove_retweets(X)
# X = remove_handles(X)
# X = remove_urls(X)
# X = reduce_lengthening(X)

fe = FeatureExtractor(X, y, [3], [ABC_LOWER], [30])
df = fe.extract_ngrams([3], [ABC_LOWER], [30])
# loaded_df = fe.load_ngrams([3], [3])
# loaded_df = fe.load_ngrams()

# df = fe.extract_unicode_blocks()
# loaded_df = fe.load_unicode_blocks()

# df = fe.extract_all()
# loaded_df = fe.load_all()


# X = extract_unicode_blocks_feature(X)
# X = pd.concat([X, extract_ngrams_feature(X, 1, ABC_LOWER+ACCENTS_LOWER, k='all')], axis=1)
# X = pd.concat([X, extract_ngrams_feature(X, 1, OTHER_SYMBOLS, k='all')], axis=1)
# X = pd.concat([X, extract_ngrams_feature(X, 2, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 1000)], axis=1)
# X = pd.concat([X, extract_ngrams_feature(X, y, 3, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 500)], axis=1)
# X = pd.concat([X, extract_ngrams_feature(X, 4, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 500)], axis=1)
# X = pd.concat([X, extract_ngrams_feature(X, 5, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 500)], axis=1)
# X = extract_ngrams_feature(X, y, 3, ABC_LOWER+ACCENTS_LOWER+BASIC_PUNCTUATION, 100)
# y, _ = y.factorize()
# train_data = lgb.Dataset(X, label=y)
#
# num_round = 10
# param = {'num_leaves': 31, 'num_trees': 150, 'objective': 'multiclass', 'num_class': 8,
#          'metrics': 'multi_error'}
#
# result = lgb.cv(param, train_data, num_round, nfold=3)
