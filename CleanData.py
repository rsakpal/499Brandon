# Cleans card data for Machine Learning input and writes it to cleaned_cards.json.
# Takes file name of json card data to be cleaned as a command line argument.
# Example call: python .\CleanData.py "AllPrintings.json"
# Data from https://mtgjson.com/api/v5/AllPrintings.json (accessed 8/4/21)

import json
import sys
import re
import os


PATH = os.path.join(os.path.dirname(__file__), 'resources', 'AllPrintings.json')

# sys.argv[1]
with open(PATH, 'r', encoding='utf-8') as f:
  all_cards = json.load(f)

seen_cards = set()
cleaned_cards = {
    'All': [],
    'Creature': [],
    'Artifact': [],
    'Enchantment': [],
    'Land': [],
    'Instant': [],
    'Sorcery': [],
    'Planeswalker': []
  }

for item in all_cards['data']:
  set = all_cards['data'][item]
  for card in set['cards']:
    if card['name'] not in seen_cards and card['borderColor'] == 'black':
      seen_cards.add(card['name'])
      clean_card = {
        'name': card['name'].lower(),
        'manaCost': card['manaCost'] if 'manaCost' in card else '',
        'convertedManaCost': card['convertedManaCost'],
        'supertypes': card['supertypes'],
        'types': card['types'],
        'subtypes': card['subtypes'],
        'type': card['type'].replace(' — ', ' ').lower(),
        'text': card['text'].lower() if 'text' in card else '',
        #'flavorText': card['flavorText'] if 'flavorText' in card else '',
        'power': card['power'] if 'power' in card else '',
        'toughness': card['toughness'] if 'toughness' in card else ''
        }

      clean_card['text'] = clean_card['text'].replace('counter target', 'uncast target')
      clean_card['text'] = clean_card['text'].replace(clean_card['name'].lower(), '@')
      clean_card['text'] = clean_card['text'].replace('\n', ' % ')
      clean_card['text'] = re.sub('(\{.*?\})', lambda match: match.group(1).upper(), clean_card['text'])
      for _ in range(4):
        if clean_card['text'] != '' and clean_card['text'][0] == '(':
          clean_card['text'] = re.sub('(\s*\(.*?\) % )', '', clean_card['text'])
      for _ in range(4):
        clean_card['text'] = re.sub('(\s*\(.*?\))', '', clean_card['text'])

      for type in clean_card['types']:
        if type in cleaned_cards.keys():
          cleaned_cards['All'].append(clean_card)
          cleaned_cards[type].append(clean_card)

with open('cleaned_cards_2.json', 'w', encoding='utf-8') as f:
  json.dump(cleaned_cards, f)