# Fine-Tuning BERT Model for Sentiment Analysis on Customer Reviews Dataset
### Pytorch implementation of finetuning bert on  CR dataset using k-fold cross validation.



This repository contains code for fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis on the Customer Reviews (CR) dataset using 10-fold cross-validation and cross-entropy loss function. The goal is to classify customer reviews into positive, negative, or neutral sentiment categories.

## Dataset
The Customer Reviews (CR) dataset is a widely used benchmark dataset in the field of natural language processing (NLP) and sentiment analysis. It consists of a large collection of customer reviews from various domains such as electronics, movies, books, and more. Each review in the dataset is accompanied by a corresponding sentiment label indicating whether the sentiment expressed in the review is positive, negative, or neutral.

## Model
The BERT model used in this project is pre-trained on a large corpus of text data and fine-tuned on the CR dataset for sentiment analysis. BERT is a powerful transformer-based model that captures contextual information from both left and right contexts of words in a sentence, making it well-suited for various NLP tasks, including sentiment analysis.

### Training Procedure
The training procedure involves 10-fold cross-validation, where the dataset is split into 10 equal-sized folds, and the model is trained and evaluated on each fold separately. Cross-entropy loss function is used as the optimization criterion, and the model's performance is evaluated based on accuracy.

## Results

|  - | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 | fold 8 | fold 9 | fold 10 |
|----------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Accuracy | 0.888594164456233 | 0.880636604774535 | 0.909090909090909 | 0.885941644562334 | 0.907161803713527 | 0.893899204244031 | 0.913419913419913 | 0.917771883289124| 0.891246684350132| 0.941644562334217 |
| F1-score | 0.913934426229508 | 0.909090909090909 | 0.925438596491228 | 0.914512922465208 | 0.927234927234927 | 0.913419913419913 | 0.947976878612716 | 0.933333333333333| 0.914760914760914| 0.955465587044534 | 




After 10-fold cross-validation, the average accuracy achieved by the fine-tuned BERT model on the CR dataset is 0.9029 and the average F1-score is 0.9255, demonstrating its effectiveness in sentiment analysis tasks.

## Requirements
Python 

PyTorch

Transformers library (from Hugging Face)

datasets (from Hugging Face)

accelerate (from Hugging Face)

scikit-learn

pandas

NumPy

## Usage
To reproduce the experiments and fine-tune the BERT model on the CR dataset, follow these steps:

1- Download jupyter notebook and upload it to [google colab](https://colab.research.google.com)

2- Go to your google drive and Create these directories recursively nlp/datasets/CR_DATASET, nlp/saved_models/cross_validation (Or you can change it in code)

3- Run all cells for first fold of cross validation then go to the ```Load each fold``` section and Run all cells after that for each fold

alternatively, you can clone the repository on your local machine & Install the required dependencies listed in the requirements section 




## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Benyormin/fineTuneBert/blob/main/LICENSE) file for details.
