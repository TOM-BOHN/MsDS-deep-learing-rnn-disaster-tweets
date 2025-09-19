"""
Text Processing Module

This module contains functions for text preprocessing, normalization,
and cleaning operations used in NLP projects.

Functions include:
- Text normalization and cleaning
- Tokenization utilities
- Stopword removal
- Stemming and lemmatization
- Text vectorization helpers

Source: Recycled from previous Natural Language Processing (NLP) analysis projects.
The general pattern of completing text processing for NLP typically includes
a comprehensive set of cleaning and normalization functions.

Original Source: auto_classifying_salesforce_cloud_documentation.ipynb
"""

from .functions import (
    advanced_text_preprocessing,
    convert_to_lowercase,
    remove_whitespace,
    remove_punctuation,
    remove_html,
    remove_emoji,
    remove_http,
    convert_acronyms,
    convert_contractions,
    remove_stopwords,
    pyspellchecker,
    text_stemmer,
    text_lemmatizer,
    discard_non_alpha,
    convert_numbers_to_words,
    keep_pos,
    remove_additional_stopwords,
    text_normalizer,
    text_normalizer_conservative,
    apply_text_normalizer
)

__all__ = [
    'advanced_text_preprocessing',
    'convert_to_lowercase',
    'remove_whitespace',
    'remove_punctuation',
    'remove_html',
    'remove_emoji',
    'remove_http',
    'convert_acronyms',
    'convert_contractions',
    'remove_stopwords',
    'pyspellchecker',
    'text_stemmer',
    'text_lemmatizer',
    'discard_non_alpha',
    'convert_numbers_to_words',
    'keep_pos',
    'remove_additional_stopwords',
    'text_normalizer',
    'text_normalizer_conservative',
    'apply_text_normalizer'
]
