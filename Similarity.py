import enum
import json
import os
import re
import warnings
import sys
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from sentence_transformers import SentenceTransformer

def space_out_punc(text):
  text = text.replace('\n', '%')
  text = re.sub('([.,():"%/+-])', r' \1 ', text)
  text = re.sub('([{])', r' \1', text)
  text = re.sub('\s{2,}', ' ', text)
  text = text.strip()
  return text

def generate_card_vocabs(cards):
  vocabs = []
  max_len = max([len(space_out_punc(card['text']).split(' ')) for card in cards])
  #print(max_len)

  for card in cards:
    vocab = []
    text = card['text']
    text = space_out_punc(text)
    words = text.split(' ')
    vocab = words
    for _ in range(max_len - len(words)):
      vocab.append('')
    vocabs.append(vocab)
  return vocabs


def calculate_cosine_score(base_document, documents):
  vectorizer = TfidfVectorizer()
  if base_document in documents:
    documents.remove(base_document)
  documents.insert(0, base_document)
  embeddings = vectorizer.fit_transform(documents)
  cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
  documents.remove(base_document)
  print(round(cosine_similarities[0], 3))

  scores = [score for _, score in enumerate(cosine_similarities)]
  max_score = max(scores)
  min_score = min(scores)
  mean_score = sum(scores) / len(scores)
  # Standard Deviation formula:
  sd_score = (sum(
      [(score - mean_score) ** 2 for score in scores]) / len(scores)
    ) ** 0.5

  # max_score = 0
  # max_score_index = 0
  # for i, score in enumerate(cosine_similarities):
  #   if max_score < score:
  #     max_score = score
  #     max_score_index = i

  # return max_score, all_texts[max_score_index]

  return round(mean_score, 3), round(sd_score, 3), round(min_score, 3), round(max_score, 3)

def calculate_use_score(card, documents):
  filename = './models/universal-sentence-encoder_4'
  model = hub.load(filename)
  base_embeddings = model([card])
  embeddings = model(documents)
  scores = cosine_similarity(base_embeddings, embeddings).flatten()
  return scores

def calcualte_bert_score(card, card_vocabs):
  model = SentenceTransformer('bert-base-nli-mean-tokens')
  sentences = sent_tokenize(card)
  base_embeddings_sentences = model.encode(sentences)
  base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)

def load_critical_words():
  critical_words = {}
  with open('keywords.txt') as f:
    words = f.read().splitlines()
    for word in words:
      critical_words.add(word)
  with open('critical_words.txt') as f:
    words = f.read().splitlines()
    for word in words:
      critical_words.add(word)
  return critical_words

def predict_score(model, input, max_len):
  input = space_out_punc(input)
  encoded = tokenizer.texts_to_sequences([input])[0]
  encoded = pad_sequences([encoded], maxlen=max_len, padding='pre')
  encoded = encoded.reshape(-1, 1, max_len)
  pred = model.predict(encoded, verbose=0)
  return pred

PATH = os.path.join(os.path.dirname(__file__), 'resources', 'all_creatures_2.txt')
with open(PATH, 'r') as f:
  documents = f.read().split('\n')

documents = ['\n'.join(documents)]

cards = [
  'S|| Legot Spirit | flying when @ enters the battlefield, you may put a creature token with flying.) | 2/2 ||E',
  'S|| Storm Sage | {2}{B} | Creature Human Wizard | flying when @ enters the battlefield, you may put a creature token with flying.) | 2/2 ||E',
  'S|| {t}: target creature gets +1/+1 until end of turn. | 1/1 ||E',
  'S|| Shaman Spirit | {2}{R}{R} | Creature Giant | flying whenever @ attacks, it gets +1/+1 until end of turn. | 1/1 ||E',
  'S|| /lan may Creature Hentawk | @ can\'t be blocked except by artifact creature. if you do to tap enters the battlefield, target creatures. | 6/6 ||E',
  'S|| Limin | {3}{W} | Creature Human Wizard | {t}: target creature gains flying until end of turn. | 1/1 ||E',
  'S|| Chopline Bird | escape—{w}: target knight permanent. | 1/1 ||E',
  'S|| of the Seaper | {1}{W} | Creature Human Vampire | whenever a creature you control attacks, it gets +1/+1 until end of turn. | 5/5 ||E',
  'S|| Us, the Racketh | {2}{G} | Creature Elf Druid | when @ enters the battlefield, if it enters the battlefield, denture that was next turn. encole a creature card from a graveyard, sacrifice @ from a graveyard. | 2/2 ||E',
  'S|| Uros, Slite Drover | {3}{W}{U}{B}{R} | Legendary Creature Human Pirate | you may have a card and you lose 2 life and draw an elster — @\'s power and toughness are each equal to the number of +1/+1 counters on it. whenever you cast a spirit or arcane spell, put a +1/+1 counter on @. | 5/5 ||E',
  'S|| forsaken drifters | {3}{B} | creature zombie | when @ dies, mill four cards. | 4/2 ||E'
]

cards = [
  's|| a an - a any the an a an - exile the exile 2 2 cards 1 2 {c} 2 2 2 2 white creature token you mana dealt phase combat . | 2 / 0 ||e',
  's|| any the a an - any a an combat end the an anywhere a any combat the and goblin warrior : this creature deals 1 2 + 1 {r} : regenerate @ deals 1 damage + 6 until end of combat . | 1 / 1 ||e',
  's|| other . - : the giant , put up to one cards an opponent owns . @ becomes a , it can\'t be blocked . % whenever a player casts an instant or sorcery spell , @ gains trample until end of turn . ( to manifest the next 1 damage that would be dealt this turn . ( exile the spell with fewer permanents from anywhere until you lose 1 life . | 3 / 4 ||e',
  's|| target . - {u} {u} | creature elemental monk | forestwalk ( this creature can\'t be blocked as long as defending player controls a forest . ) | 2 / 1 ||e',
  's|| / / / 2 / 5 | ( an opponent wins a nontoken humans get + 1 / + 1 . % whenever you cast a wall more or more more cards are less . you may put that card on the bottom of your library face exile a land and this way card tails {g} . x is more . ||e',
  's|| artifact creature elf wizard | 1 / 1 | {g} {g} , {t} : target creature gets + 2 / + 0 until end of turn . ||e',
  's|| {2} {b} {g} | creature elf cleric | 2 / 2 | {1} {r} , {t} , discard a card : @ deals x damage to target opponent destroy that creature . ||e',
  's|| {3} {w} {u} {b} {r} | legendary creature human pirate | 6 / 5 | trample % {r} : @ gets + 1 / + 1 until end of turn . ) % {b} {r} : @ gains deathtouch until end of turn . ||e'
]

vectorizer = TfidfVectorizer()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for card in cards:
      mean_score, sd_score, min_score, max_score = calculate_cosine_score(card, documents)
      print('Mean: {}, SD: {}, Min: {}, Max: {}'.format(mean_score, sd_score, min_score, max_score))
      print('Card: {}\n'.format(card))
#cards = [space_out_punc(card['text']) for card in data['Creature']]

# for card in cards[0:5]:
#   scores = calculate_use_score(card, cards)
#   print(scores)

# sys.exit()

# Generate list of words of each card
#card_vocabs = generate_card_vocabs(data['Creature'])
# print(len(card_vocabs))

# # Calculate scores
# scores = []
# for card in card_vocabs:
#   with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     score, most_similar_card = calculate_cosine_score(' '.join(card), [' '.join(card_vocab) for card_vocab in card_vocabs])
#     scores.append(score)

# df = pd.DataFrame(list(zip(scores, card_vocabs)), columns=['Score', 'Text'])
# print(df)

# data = ''
# for i in range(len(scores)):
#   data += str(scores[i]) + ' ' + ' '.join(card_vocabs[i]) + '\n'
# with open('card_scores.txt', 'w', encoding='utf-8') as f:
#   f.write(data)

# df.to_csv('card_scores.csv')

# Train on scores

# with open('card_scores.txt', 'r') as f:
#   data = f.read()
# lines = data.split('\n')[:-1]

# scores = [float(line.split(' ')[0]) for line in lines]
# texts = [(' ').join(line.split(' ')[1:]) for line in lines]
# #texts = [np.array(line.split(' ')[1:]).astype(np.str_) for line in lines]

# max_len = max([len(text) for text in texts])
# print('maxlen: ', max_len)

# #X = np.array(texts)
# y = np.array(scores).astype(np.float32)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# X = pad_sequences(sequences, maxlen=max_len)
# X = X.reshape(-1, 1, max_len)
# #X = [[tf.cast(word, dtype='float32') for word in sequence] for sequence in X]

# model = Sequential([
#   LSTM(100, input_shape=(1, max_len), return_sequences=True),
#   LSTM(100, activation='relu', return_sequences=True,),
#   Dense(10, activation='relu', input_shape=(max_len,)),
#   Dense(10, activation='relu'),
#   Dense(1,  activation='linear')
# ])
# model.compile(loss='mean_squared_error')
# model.summary()
# model.fit(X, y, batch_size=16, epochs=5)

# test_inputs = [
#   '@ enters the battlefield, each opponent loses 1 life.',
#   'when @ enters the battlefield, each opponent loses 1 life.',
#   """when @ enters the battlefield, if it enters the battlefield,
#     denture that was next turn. encole a creature card from a graveyard,
#     sacrifice @ from a graveyard.""",
#   """when @ enters the battlefield, if it enters the battlefield,
#     denture that was next turn. return a creature card from a graveyard,
#     sacrifice @ from a graveyard.""",
#   'flying',
#   'this spell can\'t be counterer, hed three 1/1 player with total power 6 or less from your graveyard to your hand.',
#   '{e/b}, {t}, discard two cards.',
#   'when @ dies , it deals 2 damage to target creature .',
#   '@ when dies , it deals damage 2 to creature target .'
#   ]

# for input in test_inputs:
#   with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     pred_score = predict_score(model, space_out_punc(input), max_len)
#     actual_score, most_similar_text = calculate_cosine_score(input, cards)
#     print('predicted:', pred_score)
#     print('actual:', actual_score)
#     print('text:', '\n', input, '\n')