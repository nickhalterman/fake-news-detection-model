import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.fact_check import fact_check

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Load data

data = pd.read_csv("data/news.csv")

# Preprocess data

data = data.drop(["Unnamed: 0"], axis=1)

# Data encoding

le = preprocessing.LabelEncoder()
le.fit(data["label"])
data["label"] = le.transform(data["label"])

# Variables

embedding_dim = 50
max_length = 54
padding_type = "post"
trunc_type = "post"
oov_tok = "<00V>"
training_size = len(data)
test_portion = 0.1

# Tokenization

# Tokenizer1.fit_on_texts(title): Fits the tokenizer on the 'title' column to create a vocabulary.
# pad_sequences(sequences1): Pads the sequences to ensure they all have the same length.

title = []
text = []
labels = []
for x in range(training_size):
    title.append(data["title"][x])
    text.append(data["text"][x])
    labels.append(data["label"][x])

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = pad_sequences(
    maxlen=max_length, sequences=sequences1, padding=padding_type, truncating=trunc_type
)

# Split Data for Training and Testing

# training_sequences1, test_sequences1: Splits the tokenized and padded data into training and testing sets.
# training_labels, test_labels: Splits the corresponding labels into training and testing labels.

split = int(test_portion * training_size)
training_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

# Reshaping Data for LSTM

training_sequences1 = np.array(training_sequences1)
test_sequences1 = np.array(test_sequences1)

# Generate Word Embeddings

embedding_index = {}
with open("data/glove/glove.6B.50d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size1 + 1, embedding_dim))

for word, i in word_index1.items():
    if i < vocab_size1:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Model Architecture

# Embedding: The embedding layer uses pre-trained GloVe embeddings.
# Conv1D: a 1D convolutional layer to detect patterns in the text.
# LSTM(64): An LSTM layer to capture long-term dependencies in the data.

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(
            vocab_size1 + 1, embedding_dim, weights=[embedding_matrix], trainable=False
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Training the Model

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    training_sequences1,
    np.array(training_labels),
    epochs=50,
    validation_data=(test_sequences1, np.array(test_labels)),
    callbacks=[early_stop],
    verbose=2,
)

headlines = [
    "Senate passes budget bill after late-night vote",
    "Obama meets with world leaders at G20 summit",
    "Federal Reserve holds interest rates steady",
    "SHOCKING: Senator caught in secret scandal the media won't report",
    "BREAKING: Leaked documents prove election was rigged",
    "You won't believe what this politician just said about voters",
    "Senate votes to make Antarctica the 51st state",
    "Shocking video exposes what the government is hiding from you.",
]

for h in headlines:
    seq = tokenizer1.texts_to_sequences([h])
    seq = pad_sequences(
        seq, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    score = model.predict(seq, verbose=0)[0][0]
    style_ok = score >= 0.5

    facts_ok, reason = fact_check(h)

    if style_ok and facts_ok:
        final = "TRUE"
    elif not style_ok and not facts_ok:
        final = "FAKE"
    else:
        final = "UNSURE"

    YELLOW = "\033[93m"
    color = GREEN if final == "TRUE" else RED if final == "FAKE" else YELLOW

    print(f"\n{color}{final}{RESET}  {h}")
    print(f"    Style robot: {'TRUE' if style_ok else 'FAKE'} ({score:.2f})")
    print(f"    Fact-checker: {reason}")
