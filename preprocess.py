import re


def reduce_lengthening(tweets):
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 3.
    """
    pattern = re.compile(r"(.)\1+")
    without_repetitions = tweets.swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(lambda t: re.sub(pattern, r'\1\1', t))
    return without_repetitions


def remove_handles(tweets):

    pattern = r'(^|[^@\w])@(\w{1,15})\b'
    without_handles = tweets.swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(lambda t: re.sub(pattern, '', t))

    return without_handles


def remove_retweets(tweets, ignore_retweets_with_no_handle=False):

    pattern = '^RT @?[\w]+: '
    if ignore_retweets_with_no_handle:
        pattern.replace('@?', '@')

    without_retweets = tweets.swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(lambda t: re.sub(pattern, '', t))
    return without_retweets


def remove_urls(tweets):
    pattern = 'http\S+'

    without_urls = tweets.swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(lambda t: re.sub(pattern, '', t))
    return without_urls
