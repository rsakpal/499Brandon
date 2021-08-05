# Removes foreign language data from the data set.
# Data from https://mtgjson.com/api/v5/AllPrintings.json

import json

f = open('AllPrintings.json', 'r', encoding='utf-8')
data = json.load(f)
f.close()

for item in data['data']:
  #print("...")
  set = data['data'][item]
  #print(set)
  for card in set['cards']:
    card['foreignData'] = []
    #print(card)

f = open('AllPrintings.json', 'w', encoding='utf-8')
json.dump(data, f)
f.close()