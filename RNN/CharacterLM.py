import os
import tensorflow as tf
#import numpy as np
from numpy import array
from pickle import dump
from pickle import load
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

# Define models
def define_model(X):
  model = Sequential()
  model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
  #model.add(LSTM(100, return_sequences=True))
  model.add(LSTM(100))
  model.add(Dense(vocab_size, activation='softmax'))
  # compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  #model.summary()
  return model

# Generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
  in_text = seed_text
  # generate a fixed number of characters
  for _ in range(n_chars):
    # encode the characters as integers
    encoded = [mapping[char] for char in in_text]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # one hot encode
    encoded = to_categorical(encoded, num_classes=len(mapping))
    # predict character

    # TODO: Add some randomization to sampling/prediction to prevent looping text
    yhat = model.predict_classes(encoded, verbose=0)
    # reverse map integer to character
    out_char = ''
    for char, index in mapping.items():
      if index == yhat:
        out_char = char
        break
    #append to input
    in_text += out_char
  return in_text

# Data Preparation

PATH = os.path.join(os.path.dirname(__file__), '..', 'resources', 'all_creatures_2.txt')
raw_text = load_doc(PATH).replace('\n', ' % ')

tokens = raw_text.split()
raw_text = ' '.join(tokens)
#print(raw_text)

length = 30
sequences = list()
for i in range(length, len(raw_text)):
  seq = raw_text[i-length:i+1]
  sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

out_filename = 'char_sequences_5.txt'
save_doc(sequences, out_filename)

# Train Language Model

# Use CPU if memory requirements are too large for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

in_filename = 'char_sequences_5.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
  encoded_seq = [mapping[char] for char in line]
  sequences.append(encoded_seq)

vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# Split input (characters 1-n) and output (character n+1)
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define the model
model = define_model(X)
# fit model
# batch size = 256?
model.fit(X, y, batch_size=32, epochs=25, validation_split=0.1, verbose=2)
# save model to file
model.save('char_model_5.h5')
# save mapping
dump(mapping, open('char_mapping_5.pk1', 'wb'))