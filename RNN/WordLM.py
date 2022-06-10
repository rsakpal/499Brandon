import json
import os
import string
import re
from tabnanny import verbose
import numpy as np
from keras.utils import version_utils
from numpy import array, pad
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def space_out_punc(text):
    text = text.replace('\n', '%')
    text = re.sub('([.,():"%/+-])', r' \1 ', text)
    text = re.sub('([{])', r' \1', text)
    text = re.sub('\s{2,}', ' ', text)
    text = text.strip()
    text = text.lower()
    return text

# define the model
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, seq_length, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

# generate a sequence from a language model
def generate_seq(model, mapping, mapping_reverse, seq_length, seed_text, max_words):
    in_text = seed_text
    prev_word = ''
    while prev_word != '||e':
        encoded = [mapping_reverse[word] for word in in_text.split()]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        predictions = model.predict(encoded, verbose=0)[0]

        #yhat = np.random.choice(len(predictions), p=predictions)
        yhat = np.argmax(predictions)
        out_word = mapping[yhat]
        prev_word = out_word
        in_text += ' ' + out_word

    return in_text

def main():
    # PREP DATA

    # PATH = os.path.join(os.path.dirname(__file__), '..', 'resources', 'all_creatures.txt')
    # raw_text = load_doc(PATH)

    # tokens = space_out_punc(raw_text).split()

    # length = 50 + 1
    # sequences = []
    # lengths = []
    # for i in range(length, len(tokens)):
    #     seq = tokens[i-length:i]
    #     lengths.append(len(seq))
    #     line = ' '.join(seq)
    #     sequences.append(line)

    # #lengths = [len(s) for s in sequences]
    # #print(lengths[:50])

    # print('Total Sequences: %d' % len(sequences))
    # # save sequences to file
    # out_filename = 'creature_word_sequences_test.txt'
    # save_doc(sequences, out_filename)

    # # TRAIN

    # # #os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    in_filename = 'creature_word_sequences_short.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')

    vocab = {}

    for line in lines:
        line = line.split()
        for word in line:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    word_mapping = {}
    word_mapping_reverse = {}
    i = 0
    for word in vocab.keys():
        word_mapping[i] = word
        word_mapping_reverse[word] = i
        i += 1

    print('Vocab Size:', len(word_mapping))

    # sequences = [[word_mapping_reverse[word] for word in line.split()] for line in lines]

    # vocab_size = len(word_mapping.keys()) + 1
    # #sequences = array(sequences)
    # #X = pad_sequences(sequences, maxlen=length-1)
    # X = np.array([np.array(seq[:-1]).astype(np.intc) for seq in sequences])
    # y = np.array([np.array([seq[-1]]).astype(np.intc) for seq in sequences])

    # print(len(X), len(y))
    # y = to_categorical(y, num_classes=vocab_size)

    # # define model
    # model = define_model(vocab_size, length-1)
    # # fit model
    # # model.fit(X, y, batch_size=256, epochs=75)
    # model.fit(X, y, batch_size=256, epochs=5, validation_split=0.1, verbose=2)

    # # model.save('word_model_short.h5')
    # # #dump(tokenizer, open('word_tokenizer.pk1', 'wb'))


    # USE MODEL

    char_model_seed_text = [
        's|| {7} | artifact creature demon |',
        's|| {2} {b} {g} | creature elf cleric |',
        's|| {2} {b} {r} | creature lizard |',
        's|| {2} {g} | creature hydra |',
        's|| {3} {w} {u} {b} {r} | legendary creature human pirate |',
    ]

    in_filename = 'creature_word_sequences_short.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')
    length = len(lines[0].split()) - 1
    model = load_model('word_model_short.h5')
    seed_text = 's||'
    for _ in range(15):
        generated = generate_seq(model, word_mapping, word_mapping_reverse, length, seed_text, 250)
        print(generated, '\n')
    # for seed in char_model_seed_text:
    #     generated = generate_seq(model, word_mapping, word_mapping_reverse, length, seed, 250)
    #     print(generated, '\n')

if __name__ == '__main__':
    main()