#! /Volumes/Environment/HeadRepo/.venv/bin/python
# coding: utf-8



import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import numpy as np
import random
import os
import sys
import subprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import warnings

warnings.simplefilter('ignore')




dir = '/Users/glebsokolov/Library/CloudStorage/GoogleDrive-yukontaf@gmail.com/My Drive/!gitNotebooks/advancedDeepLearningWithKeras'
for root, dirs, files in os.walk(dir):
    for filename in files:
        if filename.endswith('.py'): 
            path = os.path.join(root, filename)
            subprocess.run(['p2j', path])
            os.remove(path)
            




get_ipython().system('wget --no-check-certificate https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W4/misc/Laurences_generated_poetry.txt')




os.chdir('/Users/glebsokolov/HeadRepo')




get_ipython().system('gdown 1rl94aGUTRFUWze6XgTjflkR6ogcpBj9v')




corpus =  open('/Volumes/GoogleDrive/My Drive/datasets/Laurences_generated_poetry.txt').read()




from subprocess import call
call(['gdrive', 'list', '--query', "'1vAxf-D1f_YPO-xsy6hFScnSR97zQd6ep' in parents"])




foldername = ''




glove_fname = '/Volumes/GoogleDrive/My Drive/datasets/glove.6B/glove.6B.50d.txt'
def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model
glove50d = load_glove_model(glove_fname)




glove50d['yes']




df = pd.read_csv('/Volumes/GoogleDrive/My Drive/datasets/worldUniversityRankings/education_expenditure_supplementary_data.csv', engine='python')
df.head()




genre = pd.read_feather('/Volumes/GoogleDrive/My Drive/datasets/listenings/genre.feather')
genre.head()




glove = pd.read_csv('/Volumes/GoogleDrive/My Drive/datasets/glove.6B/glove.6B.50d.txt', delimiter=' ')




df.head()




df.head(10)




EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = 160000
TRAINING_SPLIT = 0.9




SENTIMENT_CSV = '/Volumes/GoogleDrive/My Drive/datasets/sentiment140.csv'
dataset = pd.read_csv(SENTIMENT_CSV, error_bad_lines=False, encoding='latin-1')
dataset.columns = ['target', 'label', 'date', 'flag', 'user', 'text']
sentences, labels = dataset.text, dataset.label




# Bundle the two lists into a single one
sentences_and_labels = list(zip(sentences, labels))

# Perform random sampling
random.seed(42)
sentences_and_labels = random.sample(sentences_and_labels, MAX_EXAMPLES)

# Unpack back into separate lists
sentences, labels = zip(*sentences_and_labels)

print(
    f"There are {len(sentences)} sentences and {len(labels)} labels after random sampling\n"
)




def train_val_split(sentences, labels, training_split):

    train_size = int(training_split * MAX_EXAMPLES)

    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]

    return train_sentences, validation_sentences, train_labels, validation_labels




train_sentences, val_sentences, train_labels, val_labels = train_val_split(
    sentences, labels, TRAINING_SPLIT)

print(f"There are {len(train_sentences)} sentences for training.\n")
print(f"There are {len(train_labels)} labels for training.\n")
print(f"There are {len(val_sentences)} sentences for validation.\n")
print(f"There are {len(val_labels)} labels for validation.")




def fit_tokenizer(train_sentences, oov_token):
    # Instantiate the Tokenizer class, passing in the correct value for oov_token
    tokenizer = Tokenizer(oov_token=oov_token)
    # Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer




# Test your function
tokenizer = fit_tokenizer(train_sentences, OOV_TOKEN)

word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)

print(f"Vocabulary contains {VOCAB_SIZE} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in
      word_index else "<OOV> token NOT included in vocabulary")
print(f"\nindex of word 'i' should be {word_index['i']}")




def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences using the correct padding, truncating and maxlen
    pad_trunc_sequences = pad_sequences(sequences,
                                        maxlen=maxlen,
                                        padding=padding,
                                        truncating=truncating)

    return pad_trunc_sequences




# Test your function
train_pad_trunc_seq = seq_pad_and_trunc(train_sentences, tokenizer, PADDING,
                                        TRUNCATING, MAXLEN)
val_pad_trunc_seq = seq_pad_and_trunc(val_sentences, tokenizer, PADDING,
                                      TRUNCATING, MAXLEN)

print(
    f"Padded and truncated training sequences have shape: {train_pad_trunc_seq.shape}\n"
)
print(
    f"Padded and truncated validation sequences have shape: {val_pad_trunc_seq.shape}"
)




train_labels = np.array(train_labels)
val_labels = np.array(val_labels)




# Define path to file containing the embeddings
GLOVE_FILE = '/Volumes/GoogleDrive/My Drive/datasets/glove.6B/glove.6B.100d.txt'

GLOVE_EMBEDDINGS = {}

# Read file and fill GLOVE_EMBEDDINGS with its contents
with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        GLOVE_EMBEDDINGS[word] = coefs




test_word = 'dog'

test_vector = GLOVE_EMBEDDINGS[test_word]

print(
    f"Vector representation of word {test_word} looks like this:\n\n{test_vector}"
)




EMBEDDINGS_MATRIX = np.zeros((VOCAB_SIZE + 1, EMBEDDING_DIM))

# Iterate all of the words in the vocabulary and if the vector representation for
# each word exists within GloVe's representations, save it in the EMBEDDINGS_MATRIX array
for word, i in word_index.items():
    embedding_vector = GLOVE_EMBEDDINGS.get(word)
    if embedding_vector is not None:
        EMBEDDINGS_MATRIX[i] = embedding_vector




l0 = tf.keras.layers.Embedding(VOCAB_SIZE + 1,
                               EMBEDDING_DIM,
                               input_length=MAXLEN,
                               weights=[EMBEDDINGS_MATRIX],
                               trainable=False)
output0 = np.reshape(l0(train_pad_trunc_seq[0]), (1, 16, 100))




l1 = tf.keras.layers.Conv1D(128, 5, activation='relu')
output1 = l1(output0)
np.shape(output1)




l2 = tf.keras.layers.GlobalMaxPooling1D()
output2 = l2(output1)
np.shape(output2)




l3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
output3 = l3(output1)
output3




embedding_dim = 16
filters = 128
kernel_size = 5
dense_dim = 6


def create_conv_model(vocab_size, embedding_dim, maxlen, embeddings_matrix):

    # Model Definition with Conv1D
    model_conv = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1,
                                  embedding_dim,
                                  input_length=maxlen,
                                  weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
        # tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.4, (128, )),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.4, (128, )),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    model_conv.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       optimizer='adam',
                       metrics=['accuracy'])
    return model_conv




model = create_conv_model(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN, EMBEDDINGS_MATRIX)

history = model.fit(train_pad_trunc_seq,
                    train_labels,
                    epochs=5,
                    validation_data=(val_pad_trunc_seq, val_labels))




import yfinance as yf




temp = pd.read_csv(
    '/Volumes/GoogleDrive/My Drive/Algorithmic Trading A-Z with Python and Machine Learning/8. +++ PART 2 Pandas for Financial Data Analysis and Introduction to OOP +++/Part2_Materials/Video_Lecture_NBs/temp.csv',
    parse_dates=['datetime'],
    index_col='datetime')




temp




list(temp.resample('D'))




ticker = ['AAPL', 'MSFT', 'DIS', 'IBM', 'KO']
stocks = yf.download(ticker,
                     start='2010-01-01',
                     end='2022-01-01',
                     interval='1d')
stocks.to_csv('stocks.csv')
stocks = pd.read_csv('stocks.csv',
                     header=[0, 1],
                     index_col=[0],
                     parse_dates=[0])
# stocks.columns = stocks.columns.to_flat_index()
stocks




close = stocks.loc[:, 'Close'].copy()




import matplotlib.pyplot as plt

plt.style.use('seaborn')




close.plot(figsize=(15, 8), fontsize=12)
plt.legend(fontsize=12)
plt.show()




close.AAPL.div(close.iloc[0, 0]).mul(100)
norm = close.div(close.iloc[0]).mul(100)




norm




norm.plot(figsize=(15, 8), fontsize=12)
plt.legend(fontsize=12)
plt.show()




aapl = close.AAPL.copy().to_frame()
aapl.head()
aapl.shift(periods=1)




aapl['DIFF2'] = aapl.diff(periods=1)
aapl['PCT_CHANGE'] = aapl.AAPL.pct_change(periods=1).mul(100)
aapl






