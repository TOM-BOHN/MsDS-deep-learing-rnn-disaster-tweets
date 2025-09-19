"""
EDA (Exploratory Data Analysis) Module

This module contains functions for exploratory data analysis in NLP projects.
Functions are designed to help analyze text data, visualize distributions, 
and understand dataset characteristics.

Source: Recycled from previous Natural Language Processing (NLP) analysis projects.
The general pattern of completing EDA for NLP typically starts with a standard 
set of tables and charts to understand the data.

Original Source: auto_classifying_salesforce_cloud_documentation.ipynb
"""

from .eda_analysis import (
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

__all__ = [
    'replace_labels',
    'create_data_by_label',
    'quick_table_details',
    'count_field',
    'shape_df_for_stacked_barchart',
    'create_single_stacked_bar',
    'viz_class_frequency',
    'viz_char_frequency',
    'viz_word_frequency',
    'viz_word_length_frequency'
]
