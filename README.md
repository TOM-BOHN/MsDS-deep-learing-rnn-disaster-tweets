# MsDS-deep-learing-rnn-disaster-tweets

## Introduction

This project focuses on developing a deep learning model to classify tweets, determining whether they pertain to real-world disasters. By leveraging a dataset of 10,000 hand-classified tweets, the primary objective was to build a recurrent neural network (RNN) capable of accurately distinguishing between disaster-related and non-disaster-related tweets. This task is crucial for emergency response organizations and news agencies that monitor social media for real-time information.

## Methodology

The project followed a structured approach, beginning with a thorough exploratory data analysis (EDA) to understand the dataset's characteristics. The data was then subjected to a conservative text normalization process to clean the tweets while preserving important contextual information.

GloVe word embeddings were utilized to convert the text data into meaningful numerical representations. Two RNN models were developed using LSTM (Long Short-Term Memory) layers:

* **Baseline Model (Model 0):** A single bidirectional LSTM layer.

* **Improved Model (Model 1):** A more complex architecture featuring dual bidirectional LSTM layers, along with regularization techniques and non-trainable embeddings to prevent overfitting.

The models were trained on a stratified split of the data, and their performance was evaluated on a validation set and through submission to the corresponding Kaggle competition.

## Results

The improved model (Model 1) demonstrated superior performance over the baseline. The key results are summarized below:

| Model | Validation Accuracy | Kaggle Public Score | F1-Score (Disaster) |
| --- | --- | --- | --- |
| **Model 0 (Baseline)** | 80.10% | 0.79528 | 0.7432 |
| **Model 1 (Improved)** | **80.56%** | **0.80110** | **0.7533** |

The best-performing model (Model 1) achieved a **validation accuracy of 80.56%** and a **public score of 0.80110 on Kaggle**. For the "disaster" class, this model achieved a **precision of 69%** and a **recall of 83%**, indicating its effectiveness in identifying actual disaster-related tweets, which is critical for the problem's context.

## Conclusion

The project successfully developed an effective RNN-based model for classifying disaster tweets. The results indicate that a more complex, regularized dual BiLSTM architecture outperforms a simpler baseline. The final model's strong performance, particularly its high recall for disaster-related tweets, makes it a valuable tool for applications in emergency response and real-time news gathering. This work underscores the potential of deep learning in natural language processing for impactful, real-world scenarios.
