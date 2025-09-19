"""
NLP Utils Package

A comprehensive package for Natural Language Processing utilities used in 
disaster tweet classification and other NLP projects.

This package contains organized modules for:
- EDA (Exploratory Data Analysis) functions
- Text processing and normalization
- Model utilities and evaluation
- Visualization helpers

Usage:
    from nlp_utils.eda import quick_table_details, viz_class_frequency
    from nlp_utils.text_processing import text_normalizer
    from nlp_utils.model_utils import learning_curve, show_metrics
"""

__version__ = "1.0.0"
__author__ = "Thomas Bohn"

# Import commonly used functions at package level for convenience
from .eda import (
    replace_labels,
    create_data_by_label,
    quick_table_details,
    count_field,
    shape_df_for_stacked_barchart,
    create_single_stacked_bar,
    viz_class_frequency,
    viz_char_frequency,
    viz_word_frequency,
    viz_word_length_frequency
)

from .text_processing import (
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
    keep_pos,
    remove_additional_stopwords,
    text_normalizer,
    apply_text_normalizer
)

__all__ = [
    # EDA functions
    'replace_labels',
    'create_data_by_label', 
    'quick_table_details',
    'count_field',
    'shape_df_for_stacked_barchart',
    'create_single_stacked_bar',
    'viz_class_frequency',
    'viz_char_frequency',
    'viz_word_frequency',
    'viz_word_length_frequency',
    # Text processing functions
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
    'keep_pos',
    'remove_additional_stopwords',
    'text_normalizer',
    'apply_text_normalizer',
]
