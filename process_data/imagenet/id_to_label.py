# -*- coding: utf-8 -*-
"""
Map label to wordnet ID
"""
import json

def map(mapping):
    """ Label to WN ID, only keeping the first label """
    with open(mapping, 'r', encoding='utf-8') as openfile:
        data = json.load(openfile)
    data = {k: v.split(", ")[0] for k, v in data.items()}
    return {f"omw-en-{k[1:]}-n": v for k, v in data.items()}


if __name__ == '__main__':
    MAPPING_I = "./data/imagenet_mapping.json"
    MAPPING_O = "./data/id_to_label.json"
    N_M = map(mapping=MAPPING_I)
    with open(MAPPING_O, 'w', encoding='utf-8') as outfile:
        json.dump(N_M, outfile, indent=4)
