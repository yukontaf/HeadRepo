import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from pprint import pprint as print
import seaborn as sns
plt.style.use('seaborn-white')
import warnings
warnings.filterwarnings('ignore')
from navec import Navec
from slovnet.model.emb import NavecEmbedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from loguru import logger
import nltk, sys
from stop_words import get_stop_words
from pymystem3 import Mystem
from string import punctuation
from loguru import logger
sns.set_palette("pastel")

path = '/Users/glebsokolov/HeadRepo/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
cosine_similarity = lambda x,y: np.dot(x, y)/(np.linalg.norm(x, 2)*np.linalg.norm(y, 2))

NUM_WORDS = 1000
EMBEDDING_DIM = 300
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8


mystem = Mystem()
russian_stopwords = get_stop_words('ru')
def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text


df = pd.read_csv('/Users/glebsokolov/HeadRepo/test_data.csv')

tokenizer = Tokenizer(num_words = NUM_WORDS, oov_token=OOV_TOKEN)
texts_combined = [preprocess_text(i) for i in df.iloc[:,3]]
texts_cleared = [i for i in texts_combined if i !='']
tokenizer.fit_on_texts(texts_cleared)
word_index = tokenizer.word_index
sequence = lambda x: tokenizer.texts_to_sequences(x.split(' '))
sequences = pad_sequences([sequence(x) for x in texts_cleared], maxlen=MAXLEN, padding=PADDING)
keyword = lambda x: [i for i in word_index if word_index[i]==x]

def make_tensor():
    replies_tensor = tf.convert_to_tensor(sequence(texts_combined[0]))
    for seq in sequences:
        output_array = np.array([])
        for word in seq:
            initial_word = navec[keyword(word)[0]] if word != 0 else np.zeros((300,))
            output_array = np.append(output_array, initial_word)
        res = pad_sequences(output_array)
    return res
make_tensor()