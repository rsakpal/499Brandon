
import json

def load_doc(filename):
  ignore_attrs = ['flavorText', 'convertedManaCost', 'types', 'subtypes', 'supertypes', 'power', 'toughness']

  text = ''
  with open(filename, 'r') as f:
    data = json.load(f)
    for card in data['Creature']:
      text += 'S||'
      for attr in card:
        if attr not in ignore_attrs:
          text += ' ' + str(card[attr].replace('\u2212', '-')) + ' |'
      text += ' ' + str(card['power']) + '/' + str(card['toughness']) + ' |'
      text += '|E '
  return text

def main():
  filename = 'cleaned_cards.json'
  text = load_doc(filename)

  with open('all_creatures.txt', 'w') as f:
    f.write(text)

if __name__ == '__main__':
  main()