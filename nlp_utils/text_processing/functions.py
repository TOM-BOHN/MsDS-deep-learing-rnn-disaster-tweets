"""
Text Processing Functions for Natural Language Processing

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

import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from spellchecker import SpellChecker
from num2words import num2words

# Initialize tokenizer and other tools
regexp = RegexpTokenizer(r"[\w']+")
stops = stopwords.words("english")
addstops = ["among", "onto", "shall", "thrice", "thus", "twice", "unto", "us", "would"]
allstops = stops + addstops
stemmer = PorterStemmer()
spacy_lemmatizer = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
spell = SpellChecker()

# Load external dictionaries
try:
    acronyms_url = 'https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/main/JSON/english_acronyms.json'
    acronyms_dict = pd.read_json(acronyms_url, typ='series')
    acronyms_list = list(acronyms_dict.keys())
except:
    acronyms_dict = {}
    acronyms_list = []

try:
    contractions_url = 'https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/main/JSON/english_contractions.json'
    contractions_dict = pd.read_json(contractions_url, typ='series')
    contractions_list = list(contractions_dict.keys())
except:
    contractions_dict = {}
    contractions_list = []

# Additional stopwords
alphabets = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
prepositions = ["about", "above", "across", "after", "against", "among", "around", "at", "before", "behind", "below", "beside", "between", "by", "down", "during", "for", "from", "in", "inside", "into", "near", "of", "off", "on", "out", "over", "through", "to", "toward", "under", "up", "with"]
prepositions_less_common = ["aboard", "along", "amid", "as", "beneath", "beyond", "but", "concerning", "considering", "despite", "except", "following", "like", "minus", "onto", "outside", "per", "plus", "regarding", "round", "since", "than", "till", "underneath", "unlike", "until", "upon", "versus", "via", "within", "without"]
coordinating_conjunctions = ["and", "but", "for", "nor", "or", "so", "and", "yet"]
correlative_conjunctions = ["both", "and", "either", "or", "neither", "nor", "not", "only", "but", "whether", "or"]
subordinating_conjunctions = ["after", "although", "as", "as if", "as long as", "as much as", "as soon as", "as though", "because", "before", "by the time", "even if", "even though", "if", "in order that", "in case", "in the event that", "lest", "now that", "once", "only", "only if", "provided that", "since", "so", "supposing", "that", "than", "though", "till", "unless", "until", "when", "whenever", "where", "whereas", "wherever", "whether or not", "while"]
others = ["ã", "å", "ì", "û", "ûªm", "ûó", "ûò", "ìñ", "ûªre", "ûªve", "ûª", "ûªs", "ûówe"]
additional_stops = alphabets + prepositions + prepositions_less_common + coordinating_conjunctions + correlative_conjunctions + subordinating_conjunctions + others


def advanced_text_preprocessing(text):
    """
    Advanced text preprocessing with additional cleaning steps.
    
    This function performs additional preprocessing steps beyond basic normalization:
    - Normalize user mentions to USER_MENTION token
    - Normalize hashtags to HASHTAG token
    - Reduce repeated characters (e.g., "sooo" -> "so")
    - Replace numbers with NUM token
    - Clean up whitespace
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text with normalized tokens
    """
    # Remove user mentions but keep the fact that there was a mention
    text = re.sub(r'@\w+', 'USER_MENTION', text)
    
    # Remove hashtags but keep the fact that there was a hashtag
    text = re.sub(r'#\w+', 'HASHTAG', text)
    
    # Normalize repeated characters (e.g., "sooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Handle numbers (replace with NUM token)
    text = re.sub(r'\d+', 'NUM', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def convert_to_lowercase(text):
    """
    Convert text to lowercase.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lowercase text
    """
    return text.lower()


def remove_whitespace(text):
    """
    Remove leading and trailing whitespaces.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with whitespaces removed
    """
    return text.strip()


def remove_punctuation(text):
    """
    Remove punctuation from text while preserving apostrophes for contractions.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with punctuation removed
    """
    punct_str = string.punctuation
    punct_str = punct_str.replace("'", "")  # discarding apostrophe from the string to keep the contractions intact
    return text.translate(str.maketrans("", "", punct_str))


def remove_html(text):
    """
    Remove HTML tags from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with HTML tags removed
    """
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    """
    Remove emojis from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with emojis removed
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_http(text):
    """
    Remove HTTP/HTTPS URLs from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with URLs removed
    """
    http = r"https?://\S+|www\.\S+"  # matching strings beginning with http (but not just "http")
    pattern = r"({})".format(http)  # creating pattern
    return re.sub(pattern, "", text)


def convert_acronyms(text):
    """
    Convert acronyms to their full forms.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with acronyms expanded
    """
    words = []
    for word in regexp.tokenize(text):
        if word in acronyms_list:
            words = words + acronyms_dict[word].split()
        else:
            words = words + word.split()

    text_converted = " ".join(words)
    return text_converted


def convert_contractions(text):
    """
    Convert contractions to their full forms.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with contractions expanded
    """
    words = []
    for word in regexp.tokenize(text):
        if word in contractions_list:
            words = words + contractions_dict[word].split()
        else:
            words = words + word.split()

    text_converted = " ".join(words)
    return text_converted


def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    return " ".join([word for word in regexp.tokenize(text) if word not in allstops])


def pyspellchecker(text):
    """
    Fix spelling errors in text using pyspellchecker.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with spelling errors corrected
    """
    word_list = regexp.tokenize(text)
    word_list_corrected = []
    for word in word_list:
        if word in spell.unknown(word_list):
            word_corrected = spell.correction(word)
            if word_corrected == None:
                word_list_corrected.append(word)
            else:
                word_list_corrected.append(word_corrected)
        else:
            word_list_corrected.append(word)
    text_corrected = " ".join(word_list_corrected)
    return text_corrected


def text_stemmer(text):
    """
    Apply stemming to text using Porter Stemmer.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with words stemmed
    """
    text_stem = " ".join([stemmer.stem(word) for word in regexp.tokenize(text)])
    return text_stem


def text_lemmatizer(text):
    """
    Apply lemmatization to text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with words lemmatized
    """
    text_spacy = " ".join([token.lemma_ for token in spacy_lemmatizer(text)])
    return text_spacy


def discard_non_alpha(text):
    """
    Remove non-alphabetic words from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with only alphabetic words
    """
    word_list_non_alpha = [word for word in regexp.tokenize(text) if word.isalpha()]
    text_non_alpha = " ".join(word_list_non_alpha)
    return text_non_alpha


def convert_numbers_to_words(text):
    """
    Convert numbers to words instead of discarding them.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with numbers converted to words
    """
    words = regexp.tokenize(text)
    converted_words = []
    
    for word in words:
        if word.isdigit():
            # Convert number to words
            try:
                word_as_number = int(word)
                if word_as_number <= 1000000:  # Reasonable limit for conversion
                    word_in_words = num2words(word_as_number)
                    converted_words.append(word_in_words)
                else:
                    # For very large numbers, keep as is
                    converted_words.append(word)
            except:
                # If conversion fails, keep original
                converted_words.append(word)
        elif word.isalpha():
            # Keep alphabetic words
            converted_words.append(word)
        # Skip non-alphanumeric words (punctuation, etc.)
    
    return " ".join(converted_words)


def keep_pos(text):
    """
    Keep only specific parts of speech from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with only specified POS tags
    """
    import nltk
    tokens = regexp.tokenize(text)
    tokens_tagged = nltk.pos_tag(tokens)
    # Keep specific POS tags: nouns, pronouns, verbs, adverbs, wh-words
    keep_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'FW', 'PRP', 'PRPS', 'RB', 'RBR', 'RBS', 
                 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WPS', 'WRB']
    keep_words = [x[0] for x in tokens_tagged if x[1] in keep_tags]
    return " ".join(keep_words)


def remove_additional_stopwords(text):
    """
    Remove additional custom stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with additional stopwords removed
    """
    return " ".join([word for word in regexp.tokenize(text) if word not in additional_stops])


def text_normalizer(text):
    """
    Apply comprehensive text normalization pipeline.
    
    This function applies a series of text cleaning and normalization steps:
    1. Advanced preprocessing - normalize mentions, hashtags, numbers
    2. Convert to lowercase
    3. Remove whitespace
    4. Remove newlines and square brackets
    5. Remove HTTP URLs
    6. Remove punctuation
    7. Remove HTML tags
    8. Remove emojis
    9. Convert acronyms
    10. Convert contractions
    11. Remove stopwords
    12. Apply lemmatization
    13. Remove non-alphabetic words
    14. Keep only specific POS tags
    15. Remove additional stopwords
    
    Args:
        text (str): Input text
        
    Returns:
        str: Fully normalized text
    """
    text = advanced_text_preprocessing(text)
    text = convert_to_lowercase(text)
    text = remove_whitespace(text)
    text = re.sub('\n', '', text)  # converting text to one line
    text = re.sub(r'\[.*?\]', '', text)  # removing square brackets
    text = remove_http(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = convert_acronyms(text)
    text = convert_contractions(text)
    text = remove_stopwords(text)
    # text = pyspellchecker(text)  # Commented out for performance
    text = text_lemmatizer(text)
    # text = text_stemmer(text)  # Commented out in favor of lemmatization
    text = discard_non_alpha(text)
    text = keep_pos(text)
    text = remove_additional_stopwords(text)
    return text


def text_normalizer_conservative(text):
    """
    Apply conservative text normalization pipeline that preserves more meaningful words.
    
    This function applies a lighter set of text cleaning and normalization steps:
    1. Convert to lowercase
    2. Remove HTTP URLs
    3. Remove HTML tags
    4. Remove emojis
    5. Remove punctuation (except apostrophes)
    6. Convert contractions
    7. Remove only basic stopwords (not additional ones)
    8. Apply lemmatization
    9. Convert numbers to words (instead of discarding)
    
    Args:
        text (str): Input text
        
    Returns:
        str: Conservatively normalized text
    """
    text = convert_to_lowercase(text)
    text = remove_whitespace(text)
    text = re.sub('\n', '', text)  # converting text to one line
    text = re.sub(r'\[.*?\]', '', text)  # removing square brackets
    text = remove_http(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = convert_contractions(text)
    # Use only basic stopwords, not the extensive additional_stops list
    text = " ".join([word for word in regexp.tokenize(text) if word not in stops])
    text = text_lemmatizer(text)
    text = convert_numbers_to_words(text)  # Convert numbers to words instead of discarding
    return text


def apply_text_normalizer(data_train, data_test):
    """
    Apply text normalization to training and test datasets.
    
    Args:
        data_train (pd.DataFrame): Training data with 'text' column
        data_test (pd.DataFrame): Test data with 'text' column
        
    Returns:
        tuple: (data_train_norm, data_test_norm, data_train, data_test)
    """
    # Implementing text normalization
    data_train_norm, data_test_norm = pd.DataFrame(), pd.DataFrame()

    data_train_norm['normalized_text'] = data_train['text'].apply(text_normalizer)
    data_test_norm['normalized_text'] = data_test['text'].apply(text_normalizer)

    # Handle label column for both training and test data
    if 'label' in data_train.columns:
        data_train_norm['label'] = data_train['label']
    else:
        data_train_norm['label'] = None  # Add as null if doesn't exist
        
    if 'label' in data_test.columns:
        data_test_norm['label'] = data_test['label']
    else:
        data_test_norm['label'] = None  # Add as null if doesn't exist

    data_train['normalized_text'] = data_train_norm['normalized_text']
    data_test['normalized_text'] = data_test_norm['normalized_text']

    print("Size of the training set:", len(data_train_norm))
    print("Size of the test set:", len(data_test_norm))
    print()
    
    # Handle label reporting for both datasets
    if 'label' in data_train.columns:
        print("Labels in training set:", data_train_norm['label'].nunique())
    else:
        print("Labels in training set: No label column (added as null)")
        
    if 'label' in data_test.columns:
        print("Labels in test set:", data_test_norm['label'].nunique())
    else:
        print("Labels in test set: No label column (added as null)")
    print()
    # Display Training Output (normalized)
    print("Sample of test set [normalized]:\n")
    print("Columns in test set [normalized]:", data_test_norm.columns, "\n")
    print(data_train_norm.head(3))
    print()
    # Display Training Input
    print("Sample of test set [data]:\n")
    print("Columns in test set [data]:", data_test.columns, "\n")
    print(data_train.head(3))
    print()

    return data_train_norm, data_test_norm, data_train, data_test
