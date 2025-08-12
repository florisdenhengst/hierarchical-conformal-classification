# -*- coding: utf-8 -*-
"""
Converting mapping from internet: https://gist.github.com/fnielsen/4a5c94eaa6dcdf29b7a62d886f540372
({index: {'id': sthg, 'label': sthg}})

to dictionary 'label' > wn id
"""
import ast
import json

# Load the file content
with open('imagenet_label_to_wordnet_synset.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

# Convert the string content to a dictionary
data = ast.literal_eval(file_content)

mappings = {x.lower(): f"omw-en-{v['id']}" for _, v in data.items() for x in v['label'].split(', ')}
with open('data/imagenet/mappings.json', 'w', encoding='utf-8') as openfile:
    json.dump(mappings, openfile, indent=4)
print(f"# of labels with mapping: {len(mappings)}")

with open('./data/results.json', 'r', encoding='utf-8') as openfile:
    results = json.load(openfile)
results = list(set(x.lower() for _, v in results.items() for x in v["set"]) )
print(f"# of unique labels in prediction sets: {len(results)}")
print("# of labels in prediction sets with no wordnet mappings: " + \
f"{len(set(results).difference(set(mappings.keys())))}")
