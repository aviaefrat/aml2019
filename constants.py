import os
import re

LANGUAGES = {'en': 0, 'es': 1, 'fr': 2, 'in': 3, 'it': 4, 'nl': 5, 'pt': 6, 'tl': 7}

ABC_UPPER = ''.join(tuple(chr(i) for i in range(ord('A'), ord('Z') + 1)))
ABC_LOWER = ABC_UPPER.lower()
ABC = ABC_UPPER + ABC_LOWER

ACCENTS_UPPER = u'ÀÁÂÃÇÈÉÊËÌÍÎÏÑÒÓÔÕÙÚÛÜŸÆŒĲ'
ACCENTS_LOWER = ACCENTS_UPPER.lower()
ACCENTS = ACCENTS_UPPER + ACCENTS_LOWER

BASIC_PUNCTUATION = '!?"\';,. '
OTHER_SYMBOLS = u'¢₡₧₲℆ºª№¿¡«»'
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUTS_DIR = os.path.join(os.getcwd(), 'reports')

ACTIONS = ['rt', 'handle', 'url', 'red_rep']
ACTIONS_LIST = [ACTIONS[:i] for i in range(1, len(ACTIONS)+1)]
FEATURE_TYPES = ['words', 'ngrams', 'all']

VOCAB_REGEX = re.compile(r""" 
        (?:\b[^\W\d_](?:[^\W\d_]|['\-])+[^\W\d_]\b) # Words with apostrophes or dashes. 
        | 
        (?:\b[^\W\d_]+\b)                           # Words without apostrophes or dashes 
        | 
        (?:\#+[\w_]+[\w\'_\-]*[\w_]+)               # Twitter hashtags 
        """, re.I | re.X)
