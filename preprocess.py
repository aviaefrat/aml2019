import os
import re

import pandas as pd
# .swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply


class PreProcessor:

    def __init__(self, tweets, order=('rt', 'handle', 'reduce_len', 'url'), reduce_n=3, dirname=None):
        self.tweets = tweets
        self.order = order
        self.reduce_n = reduce_n
        self.dirpath = dirname or os.path.join(os.getcwd(), 'processed_data')
        self._mapping = {'rt': self.remove_retweets,
                         'handle': self.remove_handles,
                         'reduce_len': self.reduce_lengthening,
                         'url': self.remove_urls}

    def preprocess(self, dirpath=None, save=True):
        dirpath = dirpath or self.dirpath
        for func_shorthand_name in self.order:
            self.tweets = self._mapping[func_shorthand_name]()
        if save:
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, self.filename())
            self.tweets.to_csv(filepath)
        return self.tweets

    def load(self, filepath=None):
        filepath = filepath or os.path.join(self.dirpath, self.filename())
        return pd.read_csv(filepath, index_col=0)

    def filename(self):
        return f"processed-{'_'.join(self.order)}.csv"

    def reduce_lengthening(self):
        """
        Replace repeated character sequences of length 3 or greater with sequences
        of length 3.
        """
        pattern = re.compile(r"(.)\1+")
        without_repetitions = self.tweets.apply(lambda t: re.sub(pattern, r'\1\1', t))
        return without_repetitions

    def remove_handles(self):

        pattern = r'(^|[^@\w])@(\w{1,15})\b'
        without_handles = self.tweets.apply(lambda t: re.sub(pattern, '', t))

        return without_handles

    def remove_retweets(self, ignore_retweets_with_no_handle=False):

        pattern = '^RT @?[\w]+: '
        if ignore_retweets_with_no_handle:
            pattern.replace('@?', '@')

        without_retweets = self.tweets.apply(lambda t: re.sub(pattern, '', t))
        return without_retweets

    def remove_urls(self):
        pattern = 'http\S+'

        without_urls = self.tweets.apply(lambda t: re.sub(pattern, '', t))
        return without_urls
