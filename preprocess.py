import re

from constants import BASIC_PUNCTUATION, OTHER_SYMBOLS
# .swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply


def preprocess(tweets, actions=('rt', 'handle', 'letter_repeat', 'url')):
    mapping = {'rt': remove_retweets,
               'handle': remove_handles,
               'url': remove_urls,
               'red_rep': reduce_repetitions}

    for action in actions:
        tweets = mapping[action](tweets)
    return tweets


def reduce_repetitions(tweets):
    def _reduce_repetitions(n):
        """
        Replace repeated pattern appearing n or more consecutive times in string,
        with the same pattern appearing exactly n consecutive times.
        """
        repeated_pattern = re.compile(f"({pattern})\\1{{{n-1},}}", re.I)
        # todo the use of (?i) subs 'Eee' with 'EE' and 'eEe' with ee. fix it to preserve original case
        return tweets.apply(lambda t: re.sub(repeated_pattern, r'\1' * n, t))

    # reduce repetitions of single letters to at most 2
    pattern = '.'
    tweets = _reduce_repetitions(2)

    # reduce repetitions of punctuation marks and spaces to at most 1
    pattern = BASIC_PUNCTUATION+OTHER_SYMBOLS
    tweets = _reduce_repetitions(1)

    # reduce repetitions of patterns of two letters to at most 2
    pattern = '.{2,2}'
    tweets = _reduce_repetitions(2)

    return tweets


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
