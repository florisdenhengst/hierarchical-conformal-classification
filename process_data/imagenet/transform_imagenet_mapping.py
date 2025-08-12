import json

in_mapping_file = "imagenet_mapping.txt"
out_mapping_file = "imagenet_mapping.json"


def format_synset(synset_key:str, wn="omw-en") -> str:
    wordnet_id = synset_key[1:]
    return f"{wn}-{wordnet_id}-n"

mapping = {}
with open(in_mapping_file) as f:
    for line in f:
        contents = line.split(" ")
        synset_key, lemmas = contents[0], contents[1:]
        synset = format_synset(synset_key)
        mapping[''.join(lemmas).split(',')[0].strip()] = synset

with open(out_mapping_file, 'w') as f:
    json.dump(mapping, f, indent=4)