import bisect
import os
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
# import swifter
# .swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import TransformerMixin

from constants import UNICODE_BLOCKS, BLOCK_STARTING_POSITIONS, BLOCK_ORDER

FEATURES_DIR = os.path.join(os.getcwd(), 'features')


class NgramExtractor(TransformerMixin):
    def __init__(self):
        self.ngrams_ = None
        self.X_ = None
        self.y_ = None
        self.case_sensitive = None
        self.col_name_modifier = None

    def fit(self, X, y, n, chars, k=10, case_sensitive=False):
        self.X_ = X
        self.y_ = y.loc[X.index]
        self.case_sensitive = case_sensitive

        if self.case_sensitive:
            self.col_name_modifier = '_cased_'
        else:
            self.X_ = self.X_.str.lower()
            self.col_name_modifier = ''

        self.ngrams_ = self.calculate_significant_ngrams(n, chars, k)

        return self

    def transform(self, X):
        def extract_char_ngrams(s):
            ngram_counts = np.zeros(len(ngram_mapping))
            for i in range(len(s) - n + 1):
                try:
                    ngram_index = ngram_mapping[s[i:i + n]]
                    ngram_counts[ngram_index] += 1
                except KeyError:
                    pass
            total_count = ngram_counts.sum()
            _ngram_proportions = (ngram_counts / total_count
                                  if total_count > 0
                                  else ngram_counts)
            return pd.Series(_ngram_proportions)

        if self.case_sensitive:
            self.X_ = X
        else:
            self.X_ = X.str.lower()
        ngram_mapping = OrderedDict(zip(self.ngrams_, range(len(self.ngrams_))))
        n = len(self.ngrams_[0])

        ngram_proportions = self.X_.apply(extract_char_ngrams)

        ngram_proportions.columns = list(ngram_mapping)
        return ngram_proportions

    def calculate_significant_ngrams(self, n, chars, k):
        lang_ngram_counts = dict()

        for lang in set(self.y_):
            lang_tweets = self.X_[self.y_ == lang]
            lang_ngram_counts[lang] = self.get_ngram_counts(lang_tweets, n, chars)

        ngram_counts = pd.DataFrame.from_dict(lang_ngram_counts, orient='index')
        ngram_counts.fillna(0, inplace=True)

        selector = SelectKBest(chi2).fit(ngram_counts, range(len(ngram_counts)))
        best_ngrams_indices = np.flip(np.argsort(selector.scores_))
        best_ngrams = list(ngram_counts.columns[best_ngrams_indices])

        if k < len(best_ngrams):
            return best_ngrams[:k]
        else:
            return best_ngrams

    @staticmethod
    def get_ngram_counts(tweets, n, chars):
        def _get_ngram_counts(s):
            i = 0
            while i <= len(s) - n:
                for j in reversed(range(0, n)):
                    if s[i + j] not in chars:
                        i += j + 1
                        break
                else:
                    ngram_counts[s[i:i + n]] += 1
                    i += 1

        ngram_counts = defaultdict(lambda: 0)
        tweets.apply(_get_ngram_counts)
        return ngram_counts


# def _get_unicode_block(c):
#     block_index = bisect.bisect(BLOCK_STARTING_POSITIONS, ord(c)) - 1
#     block_starting_point = BLOCK_STARTING_POSITIONS[block_index]
#     return UNICODE_BLOCKS[block_starting_point]
#
#
# def _get_block_counts_dict(s):
#     block_counts = defaultdict(lambda: 0)
#     for c in s:
#         block_name = _get_unicode_block(c)
#         block_counts[block_name] += 1
#     return block_counts
#
#
# def _get_block_proportions_series(s):
#     block_counts_dict = _get_block_counts_dict(s)
#     block_counts = np.zeros(len(UNICODE_BLOCKS), dtype=np.uint8)
#     for block_name, occurrences_in_block in block_counts_dict.items():
#         block_counts[BLOCK_ORDER[block_name]] = occurrences_in_block
#
#     # calculate proportions instead of absolute values
#     block_proportions = block_counts / block_counts.sum()
#
#     return pd.Series(block_proportions)
#
#
# def extract_unicode_blocks_feature(tweets):
#
#     # create a new dataframe with all unicode block names as features
#     block_proportions = tweets.apply(_get_block_proportions_series)
#     block_proportions.columns = UNICODE_BLOCKS.values()
#
#     # remove all-zero blocks
#     block_proportions = block_proportions.loc[:, (block_proportions != 0).any(axis=0)]
#
#     return block_proportions
#
#
# def extract_ngrams_feature(tweets, labels, n, chars, k, ignore_case=True):
#     significant_ngrams = calculate_significant_ngrams(tweets, labels, n, chars, k, ignore_case)
#     ngram_proportions = extract_char_ngrams(tweets, significant_ngrams, ignore_case)
#     return ngram_proportions
