from collections import defaultdict, OrderedDict
import re

import numpy as np
import pandas as pd
# import swifter
# .swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import TransformerMixin

from constants import LANGUAGES, VOCAB_REGEX, LETTERS


class VocabExtractor(TransformerMixin):

    def __init__(self):
        self.X_ = None
        self.y_ = None
        self.vocab_ = None
        self.token_pattern_ = VOCAB_REGEX

    def fit(self, X, y):
        def extract_vocab(t):
            tokens = self.token_pattern_.findall(t)
            for token in tokens:
                if token.startswith('#'):
                    hashtags.add(token)
                else:
                    words.add(token)

        self.X_ = X.str.lower()
        self.y_ = y.loc[X.index]

        vocab = dict()
        for lang in set(self.y_):
            lang_tweets = self.X_[self.y_ == lang]
            words, hashtags = set(), set()
            lang_tweets.apply(extract_vocab)
            vocab[lang] = {'words': words, 'hashtags': hashtags}

        self.vocab_ = vocab
        return self

    def transform(self, X):
        def extract_lang_token_proportions(s, mode):
            lang_token_counts = np.zeros(len(LANGUAGES))
            tokens = VOCAB_REGEX.findall(s)

            if len(tokens) == 0:
                return lang_token_counts
            for token in tokens:
                for i, lang in enumerate(LANGUAGES):
                    if token in self.vocab_[lang][mode]:
                        lang_token_counts[i] += 1
            lang_token_proportions = lang_token_counts / len(tokens)

            return pd.Series(lang_token_proportions)
        X = X.str.lower()
        word_lang_proportions = X.apply(extract_lang_token_proportions, args=('words',))
        word_lang_proportions.columns = [f'w_{lang}' for lang in LANGUAGES]

        hashtag_lang_proportions = X.apply(extract_lang_token_proportions, args=('hashtags',))
        hashtag_lang_proportions.columns = [f'h_{lang}' for lang in LANGUAGES]

        return pd.concat([word_lang_proportions, hashtag_lang_proportions], axis=1)


class NgramExtractor(TransformerMixin):
    def __init__(self):
        self.ngrams_ = None
        self.X_ = None
        self.y_ = None
        self.case_sensitive = None
        self.col_name_modifier = None
        self.simplify_punct = None

    def fit(self, X, y, n, chars, k=500, simplify_punct=True, case_sensitive=False):
        self.X_ = X
        self.y_ = y.loc[X.index]
        self.case_sensitive = case_sensitive
        self.simplify_punct = simplify_punct

        if self.case_sensitive:
            self.col_name_modifier = '_cased_'
        else:
            self.X_ = self.X_.str.lower()
            self.col_name_modifier = ''

        self.ngrams_ = self.calculate_significant_ngrams(n, chars, k)

        return self

    def transform(self, X):
        def extract_ngram_proportions(s):
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

        if self.simplify_punct:
            self.X_ = self.X_.apply(self.simplify_punctuation)

        ngram_mapping = OrderedDict(zip(self.ngrams_, range(len(self.ngrams_))))
        n = len(self.ngrams_[0])

        ngram_proportions = self.X_.apply(extract_ngram_proportions)

        ngram_proportions.columns = list(ngram_mapping)
        return ngram_proportions

    def calculate_significant_ngrams(self, n, chars, k):
        lang_ngram_counts = dict()

        for lang in set(self.y_):
            lang_tweets = self.X_[self.y_ == lang]
            if self.simplify_punct:
                lang_tweets = lang_tweets.apply(self.simplify_punctuation)
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

    @staticmethod
    def simplify_punctuation(s):
        # replace all non-letters and spaces with ' '. don't just remove all non-letters, as tweets
        # sometimes skip spaces before or after punctuation, e.g "an end.a start", and just
        # removing all punctuation will result in "an enda start".
        s = re.sub(fr"[^{LETTERS} \-\']", " ", s)

        # reduce all sequences of remaining non-letters into a single character
        s = re.sub(r"([ \-'])\1+", r'\1', s)

        # make sure every tweet starts and ends with a single space. that gives all the ngrams
        # the option to be a part of a start or an end of a word.
        if not s.startswith(' '):
            s = f' {s}'
        if not s.endswith(' '):
            s = f'{s} '

        # remove every hypen that is not preceded a-n-d followed by a letter
        s = re.sub(rf"[^{LETTERS}]\-[^{LETTERS}]|\-[^{LETTERS}]|\-[^{LETTERS}]", " ", s)

        # remove every apostrophe that is not adjacent to a letter
        s = re.sub(rf"[^{LETTERS}]'[^{LETTERS}]", " ", s)

        # reduce all newly created spaces
        s = re.sub(" +", " ", s)

        return s


def get_features(df, type_):
    if type_ == 'ngrams':
        return df.loc[:, ~(df.columns.str.startswith('w_') | df.columns.str.startswith('h_'))]
    elif type_ == 'words':
        return df.loc[:, df.columns.str.startswith('w_')]
    elif type_ == 'all':
        return df
    else:
        raise ValueError(f'invalid feature type was specified: {type_}.'
                         'available types: {ngrams, words, all}')
