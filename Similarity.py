import json

def generate_card_vocabs(cards):
  vocabs = []
  for card in cards:
    vocab = {}
    text = card['text']
    text = text.replace('\n', ' ')
    for word in text.split(' '):
      vocab[word] = vocab[word] + 1 if word in vocab else 1
    vocabs.append(vocab)
  return vocabs

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

with open('cleaned_cards.json', 'r') as f:
  data = json.load(f)

print(len(data['Creature']))
card_vocabs = generate_card_vocabs(data['Creature'])

# for vocab in card_vocabs:
#   print(vocab)