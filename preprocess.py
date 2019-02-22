from functools import partial
import re

# .swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply


def preprocess(tweets, actions=('rt', 'handle', 'letter_repeat', 'url')):
    mapping = {'rt': remove_retweets,
               'handle': remove_handles,
               'letter_repeat': partial(reduce_lengthening, n=3),
               'url': remove_urls}

    tweets = tweets.copy()
    for action in actions:
        tweets = mapping[action](tweets)
    return tweets


def reduce_lengthening(tweets, n=3, chars=None):
    """
    Replace repeated character sequences of self.reduce_n or greater with sequences
    of length self.reduce_n.
    """
    chars = f'[{chars}]' if chars is not None else '.'
    pattern = re.compile(f"({chars})\\1{{{n-1},}}")
    without_repetitions = tweets.apply(lambda t: re.sub(pattern, r'\1' * n, t))
    return without_repetitions


def remove_handles(tweets):

    pattern = r'(^|[^@\w])@(\w{1,15})\b'
    without_handles = tweets.apply(lambda t: re.sub(pattern, '', t))

    return without_handles


def remove_retweets(tweets, ignore_retweets_with_no_handle=False):

    pattern = '^RT @?[\w]+: '
    if ignore_retweets_with_no_handle:
        pattern.replace('@?', '@')

    without_retweets = tweets.apply(lambda t: re.sub(pattern, '', t))
    return without_retweets


def remove_urls(tweets):
    pattern = r'http\S+'

    without_urls = tweets.apply(lambda t: re.sub(pattern, '', t))
    return without_urls
