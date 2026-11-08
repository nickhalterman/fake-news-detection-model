import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data

data = pd.read_csv('data/news.csv')

# Preprocess data

data = data.drop(["Unnamed: 0"], axis=1)

# Data encoding

le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

# Variables

embedding_dim = 50
max_length = 54
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<00V>"
training_size = 3000
test_portion = 0.1

# Tokenization

# Tokenizer1.fit_on_texts(title): Fits the tokenizer on the 'title' column to create a vocabulary.
# pad_sequences(sequences1): Pads the sequences to ensure they all have the same length.

title = []
text = []
labels = []
for x in range(training_size):
    title.append(data['title]' [x]])
    text.append(data['text]' [x]])
    labels.append(data['label]' [x]])

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = pad_sequences(sequences1, padding=padding_type, truncating=trunc_type)

