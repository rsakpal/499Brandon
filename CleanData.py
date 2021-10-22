# Removes foreign language data from the data set.
# Data from https://mtgjson.com/api/v5/AllPrintings.json (accessed 8/4/21)

import json

f = open('AllPrintings.json', 'r', encoding='utf-8')
all_cards = json.load(f)
seen_cards = set()
cleaned_cards = {
    'Creature': [],
    'Artifact': [],
    'Enchantment': [],
    'Land': [],
    'Instant': [],
    'Sorcery': [],
    'Planeswalker': []
  }
f.close()

for item in all_cards['data']:
  set = all_cards['data'][item]
  for card in set['cards']:
    if card['name'] not in seen_cards and card['borderColor'] == 'black':
      seen_cards.add(card['name'])
      card['foreignData'] = []
      if 'text' in card:
        card['text'].replace('\u2212', '-')
      clean_card = {
        'name': card['name'],
        'manaCost': card['manaCost'] if 'manaCost' in card else '',
        'convertedManaCost': card['convertedManaCost'],
        'supertypes': card['supertypes'],
        'types': card['types'],
        'subtypes': card['subtypes'],
        'type': card['type'],
        'text': card['text'] if 'text' in card else '',
        'flavorText': card['flavorText'] if 'flavorText' in card else '',
        'power': card['power'] if 'power' in card else '',
        'toughness': card['toughness'] if 'toughness' in card else ''
        }
      for type in clean_card['types']:
        if type in cleaned_cards.keys():
          cleaned_cards[type].append(clean_card)

with open('cleaned_cards.json', 'w', encoding='utf-8') as f:
  json.dump(cleaned_cards, f)