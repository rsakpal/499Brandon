from numpy.core.numeric import outer
import tensorflow as tf
import numpy as np
import sys
from pickle import load
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Generate a sequence of characters with a language model
def generate_cards(model, mapping, seq_length, n_cards, seed_text='S|| '):
  output = ''
  for _ in range(n_cards):
    in_text = seed_text
    end = in_text[-3:]
    while end != '||E':
      # encode the characters as integers
      encoded = [mapping[char] for char in in_text]
      # truncate sequences to a fixed length
      encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
      # one hot encode
      encoded = to_categorical(encoded, num_classes=len(mapping))
      # predict character
      predictions = model.predict(encoded, verbose=0)[0]

      # Add variability
      yhat = np.random.choice(len(predictions), p=predictions)
      #yhat = model.predict_classes(encoded, verbose=0)

      # reverse map integer to character
      out_char = ''
      for char, index in mapping.items():
        if index == yhat:
          out_char = char
          break
      #append to input
      in_text += out_char
      end = in_text[-3:]
    output += in_text + '\n\n'
    print(output)
  return output

def main():
  args = sys.argv
  # load the model
  # 'model.h5', 'mapping.pk1', N
  model_filename, mapping_filename, num_cards =  args[1:]
  model = load_model(model_filename)
  # load the mapping
  mapping = load(open(mapping_filename, 'rb'))
  seq_length = 15
  print(generate_cards(model, mapping, seq_length, int(num_cards)))

if __name__ == '__main__':
  main()