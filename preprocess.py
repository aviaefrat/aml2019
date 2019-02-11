import bisect
from collections import defaultdict

import numpy as np
import pandas as pd

from data_loader import load_language_data

UNICODE_BLOCKS = {
    0x0000: 'Basic Latin',
    0x0080: 'Latin-1 Supplement',
    0x0100: 'Latin Extended-A',
    0x0180: 'Latin Extended-B',
    0x0250: 'IPA Extensions',
    0x02B0: 'Spacing Modifier Letters',
    0x0300: 'Combining Diacritical Marks',
    0x0370: 'Greek and Coptic',
    0x0400: 'Cyrillic',
    0x0500: 'Cyrillic Supplement',
    0x0530: 'Armenian',
    0x0590: 'Hebrew',
    0x0600: 'Arabic',
    0x0700: 'Syriac',
    0x0750: 'Arabic Supplement',
    0x0780: 'Thaana',
    0x07C0: 'NKo',
    0x0900: 'Devanagari',
    0x0980: 'Bengali',
    0x0A00: 'Gurmukhi',
    0x0A80: 'Gujarati',
    0x0B00: 'Oriya',
    0x0B80: 'Tamil',
    0x0C00: 'Telugu',
    0x0C80: 'Kannada',
    0x0D00: 'Malayalam',
    0x0D80: 'Sinhala',
    0x0E00: 'Thai',
    0x0E80: 'Lao',
    0x0F00: 'Tibetan',
    0x1000: 'Myanmar',
    0x10A0: 'Georgian',
    0x1100: 'Hangul Jamo',
    0x1200: 'Ethiopic',
    0x1380: 'Ethiopic Supplement',
    0x13A0: 'Cherokee',
    0x1400: 'Unified Canadian Aboriginal Syllabics',
    0x1680: 'Ogham',
    0x16A0: 'Runic',
    0x1700: 'Tagalog',
    0x1720: 'Hanunoo',
    0x1740: 'Buhid',
    0x1760: 'Tagbanwa',
    0x1780: 'Khmer',
    0x1800: 'Mongolian',
    0x1900: 'Limbu',
    0x1950: 'Tai Le',
    0x1980: 'New Tai Lue',
    0x19E0: 'Khmer Symbols',
    0x1A00: 'Buginese',
    0x1B00: 'Balinese',
    0x1B80: 'Sundanese',
    0x1C00: 'Lepcha',
    0x1C50: 'Ol Chiki',
    0x1D00: 'Phonetic Extensions',
    0x1D80: 'Phonetic Extensions Supplement',
    0x1DC0: 'Combining Diacritical Marks Supplement',
    0x1E00: 'Latin Extended Additional',
    0x1F00: 'Greek Extended',
    0x2000: 'General Punctuation',
    0x2070: 'Superscripts and Subscripts',
    0x20A0: 'Currency Symbols',
    0x20D0: 'Combining Diacritical Marks for Symbols',
    0x2100: 'Letterlike Symbols',
    0x2150: 'Number Forms',
    0x2190: 'Arrows',
    0x2200: 'Mathematical Operators',
    0x2300: 'Miscellaneous Technical',
    0x2400: 'Control Pictures',
    0x2440: 'Optical Character Recognition',
    0x2460: 'Enclosed Alphanumerics',
    0x2500: 'Box Drawing',
    0x2580: 'Block Elements',
    0x25A0: 'Geometric Shapes',
    0x2600: 'Miscellaneous Symbols',
    0x2700: 'Dingbats',
    0x27C0: 'Miscellaneous Mathematical Symbols-A',
    0x27F0: 'Supplemental Arrows-A',
    0x2800: 'Braille Patterns',
    0x2900: 'Supplemental Arrows-B',
    0x2980: 'Miscellaneous Mathematical Symbols-B',
    0x2A00: 'Supplemental Mathematical Operators',
    0x2B00: 'Miscellaneous Symbols and Arrows',
    0x2C00: 'Glagolitic',
    0x2C60: 'Latin Extended-C',
    0x2C80: 'Coptic',
    0x2D00: 'Georgian Supplement',
    0x2D30: 'Tifinagh',
    0x2D80: 'Ethiopic Extended',
    0x2DE0: 'Cyrillic Extended-A',
    0x2E00: 'Supplemental Punctuation',
    0x2E80: 'CJK Radicals Supplement',
    0x2F00: 'Kangxi Radicals',
    0x2FF0: 'Ideographic Description Characters',
    0x3000: 'CJK Symbols and Punctuation',
    0x3040: 'Hiragana',
    0x30A0: 'Katakana',
    0x3100: 'Bopomofo',
    0x3130: 'Hangul Compatibility Jamo',
    0x3190: 'Kanbun',
    0x31A0: 'Bopomofo Extended',
    0x31C0: 'CJK Strokes',
    0x31F0: 'Katakana Phonetic Extensions',
    0x3200: 'Enclosed CJK Letters and Months',
    0x3300: 'CJK Compatibility',
    0x3400: 'CJK Unified Ideographs Extension A',
    0x4DC0: 'Yijing Hexagram Symbols',
    0x4E00: 'CJK Unified Ideographs',
    0xA000: 'Yi Syllables',
    0xA490: 'Yi Radicals',
    0xA500: 'Vai',
    0xA640: 'Cyrillic Extended-B',
    0xA700: 'Modifier Tone Letters',
    0xA720: 'Latin Extended-D',
    0xA800: 'Syloti Nagri',
    0xA840: 'Phags-pa',
    0xA880: 'Saurashtra',
    0xA900: 'Kayah Li',
    0xA930: 'Rejang',
    0xAA00: 'Cham',
    0xAC00: 'Hangul Syllables',
    0xD800: 'High Surrogates',
    0xDB80: 'High Private Use Surrogates',
    0xDC00: 'Low Surrogates',
    0xE000: 'Private Use Area',
    0xF900: 'CJK Compatibility Ideographs',
    0xFB00: 'Alphabetic Presentation Forms',
    0xFB50: 'Arabic Presentation Forms-A',
    0xFE00: 'Variation Selectors',
    0xFE10: 'Vertical Forms',
    0xFE20: 'Combining Half Marks',
    0xFE30: 'CJK Compatibility Forms',
    0xFE50: 'Small Form Variants',
    0xFE70: 'Arabic Presentation Forms-B',
    0xFF00: 'Halfwidth and Fullwidth Forms',
    0xFFF0: 'Specials',
    0x10000: 'Linear B Syllabary',
    0x10080: 'Linear B Ideograms',
    0x10100: 'Aegean Numbers',
    0x10140: 'Ancient Greek Numbers',
    0x10190: 'Ancient Symbols',
    0x101D0: 'Phaistos Disc',
    0x10280: 'Lycian',
    0x102A0: 'Carian',
    0x10300: 'Old Italic',
    0x10330: 'Gothic',
    0x10380: 'Ugaritic',
    0x103A0: 'Old Persian',
    0x10400: 'Deseret',
    0x10450: 'Shavian',
    0x10480: 'Osmanya',
    0x10800: 'Cypriot Syllabary',
    0x10900: 'Phoenician',
    0x10920: 'Lydian',
    0x10A00: 'Kharoshthi',
    0x12000: 'Cuneiform',
    0x12400: 'Cuneiform Numbers and Punctuation',
    0x1D000: 'Byzantine Musical Symbols',
    0x1D100: 'Musical Symbols',
    0x1D200: 'Ancient Greek Musical Notation',
    0x1D300: 'Tai Xuan Jing Symbols',
    0x1D360: 'Counting Rod Numerals',
    0x1D400: 'Mathematical Alphanumeric Symbols',
    0x1F000: 'Mahjong Tiles',
    0x1F030: 'Domino Tiles',
    0x20000: 'CJK Unified Ideographs Extension B',
    0x2F800: 'CJK Compatibility Ideographs Supplement',
    0xE0000: 'Tags',
    0xE0100: 'Variation Selectors Supplement',
    0xF0000: 'Supplementary Private Use Area-A',
    0x100000: 'Supplementary Private Use Area-B'
}
BLOCK_STARTING_POSITIONS = sorted(UNICODE_BLOCKS)
BLOCK_ORDER = {block: index for block, index
               in zip(UNICODE_BLOCKS.values(), range(len(UNICODE_BLOCKS)))}


def _get_unicode_block(c):
    block_index = bisect.bisect(BLOCK_STARTING_POSITIONS, ord(c)) - 1
    block_starting_point = BLOCK_STARTING_POSITIONS[block_index]
    return UNICODE_BLOCKS[block_starting_point]


def get_block_counts_dict(s):
    block_counts = defaultdict(lambda: 0)
    for c in s:
        block_name = _get_unicode_block(c)
        block_counts[block_name] += 1
    return block_counts


def get_block_counts_series(s):
    block_counts_dict = get_block_counts_dict(s)
    block_counts = np.zeros(len(UNICODE_BLOCKS), dtype=np.uint8)
    for block_name, occurrences_in_block in block_counts_dict.items():
        block_counts[BLOCK_ORDER[block_name]] = occurrences_in_block
    return pd.Series(block_counts)


def create_unicode_block_counts_feature(df, proportions=True):

    # create a new dataframe with all block counts as features
    block_counts = df['tweet_text'].apply(get_block_counts_series)
    block_counts.columns = UNICODE_BLOCKS.values()

    # remove all-zero blocks
    block_counts = block_counts.loc[:, (block_counts != 0).any(axis=0)]

    # calculated proportions instead of absolute values
    if proportions:
        block_counts = block_counts.apply(lambda x: x / x.sum(), axis=1)
    return block_counts





# def create_char_vocab(df):
#     char_vocab = set()
#     df['tweet_text'].apply(lambda x: char_vocab.update(x))
#     return char_vocab






df = load_language_data('en')
create_unicode_block_counts_feature(df)