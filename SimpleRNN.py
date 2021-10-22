
import json

def load_doc(filename):
  text = ''
  with open(filename, 'r') as f:
    data = json.load(f)
    for card in data['Creature']:
      text += card['text'].replace('\n', ' ') + '\n'
  return text

def main():
  filename = 'cleaned_cards.json'
  text = load_doc(filename)
  print(text)

if __name__ == '__main__':
  main()