#! /Volumes/Environment/HeadRepo/.venv/bin/python
# coding: utf-8



import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
from tensorflow import keras
tf.config.run_functions_eagerly(True)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adagrad, Adadelta, Adam, Adamax, Ftrl, Nadam, SGD, RMSprop 
from tensorflow.keras.layers import Bidirectional, Dense, LSTM, GRU, Conv1D, GlobalAveragePooling1D, Lambda, Flatten, Dropout, Embedding, Input
from tqdm import tqdm
matplotlib.style.use("seaborn-whitegrid")
pd.set_option("display.width", 5000)
pd.set_option("display.max_columns", 60)
plt.rcParams["figure.figsize"] = (15, 10)

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.simplefilter('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# AUTO = tf.data.experimental.AUTOTUNE




get_ipython().run_cell_magic('capture', '', '!pip install optuna\n!pip install blackcellmagic\n!pip install ipdb\n!apt install --allow-change-held-packages libcudnn8=8.4.1.50-1+cuda11.6\n')




import optuna
import ipdb
from optuna.trial import TrialState
get_ipython().run_line_magic('load_ext', 'blackcellmagic')




get_ipython().system('gdown 12ceraaz41xJ503VhZlU-WnBB5pR8QWWf')
get_ipython().system('gdown 1mocZNvYWzWL9U-kygm9QU4ejuJoO0jyo')




df = pd.read_feather('/content/train.feather')
df = df.drop('data_source', axis=1)
df = df.fillna(0)
train = df.sample(frac = 0.8)
val = df.drop(train.index, axis=0)




def split_seq(df):
    sentences = np.array([
    df[['protein_sequence']].to_numpy()[:, 0],
])
    splitted= []
    for i in sentences[0, :]:
        splitted.append(list(i))

    return np.array(splitted)

train_seq, test_seq = split_seq(train), split_seq(val)

train_ph, val_ph = train['pH'].to_numpy().reshape((len(train), 1)), val['pH'].to_numpy().reshape((len(val), 1))
train_tm, val_tm = train['tm'].to_numpy().reshape((len(train), 1)), val['tm'].to_numpy().reshape((len(val), 1))




df['protein_sequence'].apply(lambda x: len(x)).quantile(0.99)




max_length = 2250
trunc_type='post'
embedding_dim = 64




tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_seq)
sequences = tokenizer.texts_to_sequences(train_seq)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
vocab_size = len(tokenizer.word_index)
train = tf.data.Dataset.from_tensor_slices(np.append(np.append(padded, train_ph, 1), train_tm, 1))

testing_sequences = tokenizer.texts_to_sequences(test_seq)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
val = tf.data.Dataset.from_tensor_slices(np.append(np.append(testing_padded, val_ph, 1), val_tm, 1))




BATCH_SIZE = 32
def preprocess(data):
    sequence = data[:-1]
    ph = tf.reshape(data[-2], (-1, ))
    tm =  data[-1]
    return sequence, ph, tm
def get_training_dataset(dataset):
    dataset = dataset.map(preprocess).shuffle(len(sequences)).batch(BATCH_SIZE)
    return dataset
def get_validation_dataset(valid):
  valid = valid.map(preprocess).shuffle(len(sequences)).batch(BATCH_SIZE)
  return valid
def concat(input):
  # if len(input[0].shape) >=2:
  #   input[1] = tf.tile(tf.reshape(input[1], (-1, 1, 1)), (1, 1, input[0].shape[-1]))
  #   return tf.keras.layers.concatenate([input[0], input[1]], axis=1)
  # else:
    return tf.keras.layers.concatenate([input[0], input[1]], axis=1)




# %pdb
class MyModel(tf.keras.Model):
  def __init__(self, lstm_layers, emb_dim, lstm_units, dropout_rate):
    super(MyModel, self).__init__()
    self.lstm_layers, self.emb_dim, self.lstm_units, self.dropout_rate = lstm_layers, emb_dim, lstm_units, dropout_rate
    self.input_ph = tf.keras.layers.Input((1,))
    self.input_seq = tf.keras.layers.Input((500, ))
    self.seq_layers =  [] 
    self.seq_layers.extend([tf.keras.layers.Embedding(21, emb_dim, name='embedding'),
                        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
                        tf.keras.layers.Dropout(dropout_rate),  
                        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                        tf.keras.layers.Dropout(dropout_rate),
                        *[tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(i, return_sequences=True, name=f'LSTM_{i}')) for j in range(lstm_layers-1) for i in lstm_units],
                        ])
    for i in range(len(self.seq_layers)):
      vars(self)[f'SEQ_LAYER_{i}'] = self.seq_layers[i]

    self.ph_layers = [
                      tf.keras.layers.Dense(512, name='dense1_ph', activation='relu'),
                      tf.keras.layers.Dropout(dropout_rate), 
                      tf.keras.layers.Dense(256, name='dense2_ph', activation='relu'),
                      tf.keras.layers.Dropout(dropout_rate), 
                      tf.keras.layers.Dense(128, name='dense3_ph', activation='relu'),
                      # tf.keras.layers.Dropout(dropout_rate),  
                      # tf.keras.layers.Dense(1, name='output_ph')
                      ]

    self.lambda_layer = tf.keras.layers.Lambda(function=concat, name='lambda_layer')
    self.flatten = tf.keras.layers.Flatten(name='flatten')
    self.dense_combined = tf.keras.layers.Dense(64, activation='relu', name='dense_combined')
    self.lambda_helper = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 64, 1)))
    self.lstm_dense = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512))
    self.last_dense = tf.keras.layers.Dense(1, name='output')

  def call(self, inputs):
    seq, ph = inputs
    SEQ_LAYER_0 = vars(self)['SEQ_LAYER_0']
    x = SEQ_LAYER_0(seq)
    for i in range(1, self.lstm_layers):
        SEQ_LAYER_i = vars(self)[f'SEQ_LAYER_{i}']
        x = SEQ_LAYER_i(x)

    # for layer in self.seq_layers:
    #   seq = layer(seq)

    for layer in self.ph_layers:
      ph = layer(ph)

    x = self.lambda_layer([x, tf.tile(tf.reshape(ph, (-1, 128, 1)), (1, 1, x.shape[-1]))])
    x = self.flatten(x)
    x = self.dense_combined(x)
    x = self.lambda_helper(x)
    # x = self.lstm_dense(x)
    x = self.last_dense(x)
    return x
    




TRAIN_STEPS = 15
PRUNING_INTERVAL_STEPS = 50
def objective(trial):    
  lstm_layers = trial.suggest_int('lstm_layers', 1, 7)
  emb_dim = trial.suggest_int('emb_dim', 256, 1024)
  lstm_units = []
  learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1)
  for i in range(lstm_layers):
    lstm_units.append(trial.suggest_int(f'lstm_units_l{i}', 16, 512))
  dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1)
  optimizer = trial.suggest_categorical('optimizer', [Adagrad, Adadelta, Adam, Adamax, Ftrl, Nadam, SGD, RMSprop])
  regressor = MyModel(lstm_layers, emb_dim, lstm_units, dropout_rate)
  loss_obj = tf.keras.metrics.MeanAbsoluteError(name='loss_obj')
  regressor.compile(loss=loss_obj, optimizer=optimizer(learning_rate=learning_rate))
  losses, n_train_iter, step = [], len(get_training_dataset(train)), 0
  for epoch in range(1):
    print(f'Epoch # {epoch} started')
    for batch in tqdm(get_training_dataset(train)):
      predictions = regressor([batch[0], batch[1]], training=True)
      loss = loss_obj(batch[2], predictions)
      losses.append(loss)
      if step > n_train_iter//2:
        intermediate_value = loss
        if intermediate_value < best_loss:
            raise optuna.TrialPruned()
      step += 1
      best_loss = min(losses)
    print(f'Training Loss {loss:.2f}, Best Loss: {best_loss:.2f}')
    for val_batch in tqdm(get_validation_dataset(val)):
      predictions = regressor([val_batch[0], val_batch[1]], training=False)
      val_loss = loss_obj(val_batch[2], predictions)
    print(f'Validation Loss {val_loss:.2f}')
  return val_loss




# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)
# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])




# print("Best trial:")
# trial = study.best_trial

# trial




for i in get_training_dataset(train).take(1):
  x = Embedding(vocab_size + 1, 2250)(i[0])
  x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
  x = Dropout(0.3)(x)
  x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
  x = Bidirectional(LSTM(32, return_sequences=True))(x)
  x = GlobalAveragePooling1D()(x)
  x




from unicodedata import bidirectional
# %pdb
class MyModel2(tf.keras.Model):
  def __init__(self):
    super(MyModel2, self).__init__()
    self.emb_dim, self.dropout_rate = 64, 0.2
    self.input_ph = tf.keras.layers.Input((1,))
    self.input_seq = tf.keras.layers.Input((64, ))
    self.seq_layers =  [] 
    self.seq_layers.extend([
                        Embedding(vocab_size + 1, self.emb_dim, name='embedding'),
                        # Conv1D(filters=64, kernel_size=5, activation='relu'),
                        # Dropout(self.dropout_rate),  
                        Bidirectional(LSTM(128, return_sequences=True, name='LSTM_5')),
                        Dropout(self.dropout_rate),
                        Bidirectional(LSTM(32, return_sequences=True, name='LSTM_7')),
                        Dropout(self.dropout_rate),
                        Bidirectional(LSTM(16, return_sequences=True)),
                        Bidirectional(LSTM(8)),
                        # GlobalAveragePooling1D(),
                        ])

    self.ph_layers = [ 
                      tf.keras.layers.Dense(32, name='dense3_ph', activation='relu'),
                      tf.keras.layers.Dropout(self.dropout_rate),  
                      tf.keras.layers.Dense(1, name='output_ph')
                      ]

    self.lambda_layer = Lambda(function=concat, name='lambda_layer')
    self.flatten = Flatten(name='flatten')
    self.dense_combined = Dense(32, activation='relu', name='dense_combined')
    self.lambda_helper = Lambda(lambda x: tf.reshape(x, (-1, 32, 1)))
    self.last_dense = Dense(1, activation='relu', name='output')

  def call(self, inputs):
    seq, ph = inputs

    for layer in self.seq_layers:
      seq = layer(seq)

    for layer in self.ph_layers:
      ph = layer(ph)

    x = self.lambda_layer([seq, ph])
    x = self.flatten(x)
    x = self.dense_combined(x)
    x = self.lambda_helper(x)
    x = self.last_dense(x)
    return x




model = MyModel2()
adam, sgd, rms = Adam(1e-3), SGD(1e-4), RMSprop(1e-4)
loss_fn = MeanAbsoluteError()
model.compile(optimizer=adam, loss=loss_fn)




# %pdb
for epoch in range(10):
  print(f'Epoch # {epoch} started')
  for batch in tqdm(get_training_dataset(train)):
    predictions = model([batch[0], batch[1]], training=True)
    loss = loss_fn(batch[2], predictions)
    # print(tf.math.reduce_mean(loss))
  for val_batch in tqdm(get_validation_dataset(val)):
    predictions = model([val_batch[0], val_batch[1]], training=False)
    val_loss = loss_fn(val_batch[2], predictions)
  print(f'Validation Loss {tf.math.reduce_mean(val_loss):.2f}')






