import re

# .swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply


class PreProcessor:

    def __init__(self, tweets, order=('rt', 'handle', 'reduce_len', 'url'), reduce_n=3):
        self.tweets = tweets
        self.order = order
        self.reduce_n = reduce_n

        self._mapping = {'rt': self.remove_retweets,
                         'handle': self.remove_handles,
                         'reduce_len': self.reduce_lengthening,
                         'url': self.remove_urls}

    def preprocess(self):
        for func_shorthand_name in self.order:
            self.tweets = self._mapping[func_shorthand_name]()
        return self.tweets

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
