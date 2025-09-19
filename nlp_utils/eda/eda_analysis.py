"""
EDA Functions for Natural Language Processing Analysis

This module contains reusable functions for exploratory data analysis (EDA) 
in natural language processing projects. These functions are designed to help
analyze text data, visualize distributions, and understand dataset characteristics.

Source: Recycled from previous Natural Language Processing (NLP) analysis projects.
The general pattern of completing EDA for NLP typically starts with a standard 
set of tables and charts to understand the data.

Original Source: auto_classifying_salesforce_cloud_documentation.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def replace_labels(labels, label_dict):
    """
    Replace labels in a list based on a label dictionary.
    
    Args:
        labels (list): List of labels to replace
        label_dict (dict): Dictionary mapping old labels to new labels
        
    Returns:
        list: List of replaced labels
    """
    new_labels = []
    # Iterate through each label in the input list
    for label in labels:
        # If the label exists in the label dictionary, append the replacement label
        if label in label_dict:
            new_labels.append(label_dict[label])
        else:
            new_labels.append(label)

    return new_labels


def create_data_by_label(data, label_col):
    """
    Create a dictionary to store the data for each label.
    
    Args:
        data (pd.DataFrame): Input dataframe
        label_col (str): Name of the label column
        
    Returns:
        dict: Dictionary with labels as keys and corresponding dataframes as values
    """
    data_by_label = {}
    # Iterate over the unique labels
    for label in data[label_col].unique():
        # Subset the data for the current label
        data_by_label[label] = data[data[label_col] == label]

    print("Size of the data by label dictionary:", len(data_by_label), "\n")

    return data_by_label


def quick_table_details(df_name, df, level_of_detail=10):
    """
    Print key table details based on level of detail.
    
    Args:
        df_name (str): Name of the dataframe for display
        df (pd.DataFrame): Input dataframe
        level_of_detail (int): Level of detail to display (1-5)
    """
    # describe the shape and column summary
    if level_of_detail >= 1:
        print('\n####', df_name, '####')
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        print('number of features (columns) = ' + str(num_cols))
        print('number of observations (rows) = ' + str(num_rows))
        print('----------------------------', '\n')
    # print the datatype counts
    if level_of_detail >= 2:
        print('DataType Counts:')
        print(df.dtypes.value_counts())
        print('----------------------------', '\n')
    # print a full list of column names
    if level_of_detail >= 3:
        print('Columns:')
        print(df.columns)
        print('----------------------------', '\n')
    #  expanded table details
    if level_of_detail >= 4:
        print('Description:')
        print(df.describe(include='all'))
        print('----------------------------', '\n')
        print('Info:')
        print(df.info())
        print('----------------------------', '\n')
    #  table records preview
    if level_of_detail >= 5:
        print('Table Preview:')
        x_records = 3
        print(df.head(x_records))
        print('....')
        print(df.tail(x_records))
        print('----------------------------', '\n')


def count_field(df, field='label'):
    """
    Count the number of records in each category and stats.
    
    Args:
        df (pd.DataFrame): Input dataframe
        field (str): Name of the field to count
        
    Returns:
        pd.DataFrame: DataFrame with counts and percentages
    """
    df_cat_count = pd.DataFrame(df[field].value_counts()).reset_index()
    df_cat_count = df_cat_count.sort_values(by=['count'], ascending=False)
    df_cat_count['Pct of Total'] = round(df_cat_count['count'] / df_cat_count['count'].sum(), 2)
    df_cat_count['Pct of Total Text'] = (df_cat_count['Pct of Total'] * 100).apply(int).apply(str) + ' %'
    return df_cat_count


def shape_df_for_stacked_barchart(df, group_by, stack_by, normalize=True):
    """
    Define a function to aggregate dataframe on a single category.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_by (str): Column to group by
        stack_by (str): Column to stack by
        normalize (bool): Whether to normalize the values
        
    Returns:
        pd.DataFrame: Shaped dataframe for stacked bar chart
    """
    # Check if group_by and stack_by are the same
    if group_by == stack_by:
        # If they are the same, simply return the value counts as a DataFrame
        df_chart = df[group_by].value_counts(normalize=normalize).round(2).to_frame()
        df_chart.index.name = stack_by  # Set the index name for consistency
    else:
        # If they are different, proceed with the original logic
        df_chart = (df
                    .groupby(group_by)[stack_by]
                    .value_counts(normalize=normalize)
                    .round(2)
                    .unstack())
    return df_chart


def create_single_stacked_bar(df, group_by, stack_by, fig_size=(5, 5), normalize=True, gDEBUG=True):
    """
    Define a function to plot a bar chart for a single category.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_by (str): Column to group by
        stack_by (str): Column to stack by
        fig_size (tuple): Figure size
        normalize (bool): Whether to normalize the values
        gDEBUG (bool): Debug flag for printing
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    df_chart = shape_df_for_stacked_barchart(df=df, group_by=group_by, stack_by=stack_by, normalize=normalize)
    if gDEBUG: 
        print(df_chart, '\n')
    fig, ax = plt.subplots(figsize=fig_size)
    # plot the chart
    chart = df_chart.plot.bar(stacked=True,
                              ax=ax,
                              width=0.3,
                              edgecolor="black")
    # Customising legend
    ax.legend(fontsize=10, title_fontsize=10)

    ax.set_title(group_by, fontsize=10)
    ax.set_xlabel(group_by, fontsize=8)
    ax.set_ylabel('Record Count', fontsize=8)
    return fig


def viz_class_frequency(data_by_label, label_dict):
    """
    Visualization of class frequencies.
    
    Args:
        data_by_label (dict): Dictionary with data by label
        label_dict (dict): Dictionary for label replacement
    """
    values = []
    labels = []
    for i in range(0, len(data_by_label)):
        try:
            values.append(len(data_by_label[i]))
            labels.append(i)
        except:
            pass

    labels = replace_labels(labels, label_dict=label_dict)

    print(values)
    print(labels)

    # Create the horizontal bar chart
    plt.barh(labels, values)

    # Add labels and title
    plt.xlabel("Record Count")
    plt.ylabel("Labels")
    plt.title("Comparison of Class Frequencies")

    # Display the chart
    plt.show()


def viz_char_frequency(data_by_label, label_dict):
    """
    Distribution of number of characters in description.
    
    Args:
        data_by_label (dict): Dictionary with data by label
        label_dict (dict): Dictionary for label replacement
    """
    values = []
    labels = []
    for i in range(0, len(data_by_label)):
        try:
            values.append(data_by_label[i]['text'].str.len())
            labels.append(i)
        except:
            pass

    labels = replace_labels(labels, label_dict=label_dict)

    cols = 3
    rows = math.ceil(len(values)/cols)

    fig, axs = plt.subplots(rows, cols, figsize=(10, (rows*3)), sharey=False)

    # Flatten the axs array to iterate over it easily
    axs = axs.flatten()

    for i in range(0, len(data_by_label)):
        sns.histplot(x=values[i], bins=20, ax=axs[i]).set_title('Class: ' + labels[i], fontsize=10)

    fig.suptitle("Distribution of number of characters in description", y=1.05)

    for i in range(0, len(data_by_label)):
        axs[i].set_xlabel(" ") if i // cols == 0 else axs[i].set_xlabel("Number of characters")
        if i % cols != 0: 
            axs[i].set_ylabel(" ")


def viz_word_frequency(data_by_label, label_dict):
    """
    Distribution of number of words in description.
    
    Args:
        data_by_label (dict): Dictionary with data by label
        label_dict (dict): Dictionary for label replacement
    """
    values = []
    labels = []
    for i in range(0, len(data_by_label)):
        try:
            values.append(data_by_label[i]['text'].str.split().map(lambda x: len(x)))
            labels.append(i)
        except:
            pass

    labels = replace_labels(labels, label_dict=label_dict)

    cols = 3
    rows = math.ceil(len(values)/cols)

    fig, axs = plt.subplots(rows, cols, figsize=(10, (rows*3)), sharey=False)

    # Flatten the axs array to iterate over it easily
    axs = axs.flatten()

    for i in range(0, len(data_by_label)):
        sns.histplot(x=values[i], bins=20, ax=axs[i]).set_title('Class: ' + labels[i], fontsize=10)

    fig.suptitle("Distribution of number of words in description", y=1.05)

    for i in range(0, len(data_by_label)):
        axs[i].set_xlabel(" ") if i // cols == 0 else axs[i].set_xlabel("Number of words")
        if i % cols != 0: 
            axs[i].set_ylabel(" ")


def viz_word_length_frequency(data_by_label, label_dict):
    """
    Distribution of number of words in description.
    
    Args:
        data_by_label (dict): Dictionary with data by label
        label_dict (dict): Dictionary for label replacement
    """
    values = []
    labels = []
    for i in range(0, len(data_by_label)):
        try:
            values.append(data_by_label[i]['text'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x)))
            labels.append(i)
        except:
            pass

    labels = replace_labels(labels, label_dict=label_dict)

    cols = 3
    rows = math.ceil(len(values)/cols)

    fig, axs = plt.subplots(rows, cols, figsize=(10, (rows*3)), sharey=False)

    # Flatten the axs array to iterate over it easily
    axs = axs.flatten()

    for i in range(0, len(data_by_label)):
        sns.histplot(x=values[i], bins=20, ax=axs[i]).set_title('Class: ' + labels[i], fontsize=10)

    fig.suptitle("Distribution of average word-length in description", y=1.05)

    for i in range(0, len(data_by_label)):
        axs[i].set_xlabel(" ") if i // cols == 0 else axs[i].set_xlabel("Average word-length")
        if i % cols != 0: 
            axs[i].set_ylabel(" ")
