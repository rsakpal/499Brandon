
import json
import os

def load_doc(filename):
  ignore_attrs = ['flavorText', 'convertedManaCost', 'types', 'subtypes', 'supertypes', 'power', 'toughness']

  text = ''
  with open(filename, 'r') as f:
    data = json.load(f)

    print(data['Creature'][0])

    for card in data['Creature']:
      text += 'S|| '
      text += str(card['name'].replace('\u2212', '~').replace('-', '~')) + ' | '
      text += str(card['manaCost'].replace('\u2212', '~')) + ' | '
      text += str(card['type'].replace('\u2212', '~').replace('-', '~')) + ' | '
      text += str(card['text'].replace('\u2212', '~')) + ' | '
      text += str(card['power']) + '/' + str(card['toughness'])
      text += ' ||E\n'
      # for attr in card:
      #   if attr not in ignore_attrs:
      #     text += ' ' + str(card[attr].replace('\u2212', '-')) + ' |'
      # text += ' ' + str(card['power']) + '/' + str(card['toughness']) + ' |'
      # text += '|E '
  return text[:-2]

def main():
  filename = 'cleaned_cards_2.json'
  PATH = os.path.join(os.path.dirname(__file__), 'resources', filename)
  text = load_doc(PATH)

  filename = 'all_creatures_2.txt'
  PATH = os.path.join(os.path.dirname(__file__), 'resources', filename)
  with open(PATH, 'w') as f:
    f.write(text)

if __name__ == '__main__':
  main()