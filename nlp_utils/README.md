# NLP Utils Package

A comprehensive Python package for Natural Language Processing utilities used in disaster tweet classification and other NLP projects.

## Package Structure

```
nlp_utils/
├── __init__.py                 # Main package initialization
├── README.md                   # This file
├── eda/                        # Exploratory Data Analysis
│   ├── __init__.py
│   └── functions.py           # EDA functions
├── text_processing/            # Text preprocessing utilities
│   └── __init__.py
├── model_utils/               # Model training and evaluation
│   └── __init__.py
└── visualization/             # Advanced visualization tools
    └── __init__.py
```

## Usage

### Import from main package (recommended)
```python
from nlp_utils import quick_table_details, viz_class_frequency
```

### Import from specific subpackages
```python
from nlp_utils.eda import quick_table_details, count_field
from nlp_utils.text_processing import text_normalizer  # Future
from nlp_utils.model_utils import learning_curve      # Future
```

## Available Functions

### EDA Module (`nlp_utils.eda`)

- `replace_labels()` - Replace labels in a list based on a dictionary
- `create_data_by_label()` - Create dictionary with data grouped by labels
- `quick_table_details()` - Print detailed table information
- `count_field()` - Count records in each category with statistics
- `shape_df_for_stacked_barchart()` - Prepare data for stacked bar charts
- `create_single_stacked_bar()` - Create stacked bar chart visualizations
- `viz_class_frequency()` - Visualize class frequency distributions
- `viz_char_frequency()` - Visualize character count distributions
- `viz_word_frequency()` - Visualize word count distributions
- `viz_word_length_frequency()` - Visualize average word length distributions

## Future Modules

### Text Processing Module
Will contain functions for:
- Text normalization and cleaning
- Tokenization utilities
- Stopword removal
- Stemming and lemmatization
- Text vectorization helpers

### Model Utils Module
Will contain functions for:
- Learning curve visualization
- Model evaluation metrics
- Training history analysis
- Model comparison utilities
- Hyperparameter tuning helpers

### Visualization Module
Will contain functions for:
- Advanced plotting utilities
- Custom chart types
- Interactive visualizations
- Report generation helpers

## Adding New Functions

To add new functions to existing modules:

1. Add the function to the appropriate `functions.py` file
2. Import the function in the module's `__init__.py`
3. Add the function to the main package's `__init__.py` if it should be available at the top level
4. Update this README with the new function documentation

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- math (built-in)

## Version

Current version: 1.0.0

## Author

Thomas Bohn
