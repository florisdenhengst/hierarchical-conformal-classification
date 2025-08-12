import json
import wn
import ast
import functools
import numpy as np
import pandas as pd
import pickle

# Load the file content
with open('data/imagenet/imagenet_label_to_wordnet_synset.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

# Convert the string content to a dictionary
data = ast.literal_eval(file_content)
id_to_label_classif = {f"omw-en-{info['id']}": f"label_{k}" for k, info in data.items()}

def get_lca(synsets: list[wn.Synset]) -> wn.Synset:
    """ Find Lowest Common Ancestor of a set of synsets.
    The lowest common ancestor or least common subsumer is the node of highest depth of all nodes that
    have all nodes in ``synsets`` as descendant.

    Returns the node if ``synsets`` has only one element. """
    if len(synsets) == 1:
        # the lowest common ancestor is the node itself
        return synsets[0]
    else:
        # if len(ps) == 2 -> one iteration
        # if len(ps) ==3 -> find lca-1 of first two elements, then lca(lca-1, 3rd elements)
        # then recursively
        pending_synsets = synsets[2:]
        lca = wn.taxonomy.lowest_common_hypernyms(synsets[0], synsets[1], simulate_root=True)[0]
        while pending_synsets:
            synset = pending_synsets.pop()
            lca = wn.taxonomy.lowest_common_hypernyms(lca, synset, simulate_root=True)[0]
        return lca

with open("data/imagenet/mappings.json") as openfile:
    mappings = json.load(openfile)

def vanilla_proba(id_sample, ps):
    df = pd.read_csv("data/imagenet/0.05/probas.csv")
    row = df[df["id"] == id_sample]

    print({x: float(df[df["id"] == id_sample][id_to_label_classif[mappings[x]]].values[0]) for x in ps})
    # cols_ps = [id_to_label_classif[mappings[x]] for x in ps]
    # print(cols_ps)
    # print(ps)
    # print(df[df["id"] == id_sample][cols_ps].values)
    # values = df[df["id"] == id_sample][[col for col in df.columns if col.startswith("label_")]].values
    # print(np.sum(values))
    # print(np.argmax(values))

vanilla_cp = {'mouse', 'tabby', 'refrigerator', 'screen', 'printer', 'washbasin', 'tiger cat', 'radiator', 'egyptian cat', 'space heater'}
vanilla_cp = list(vanilla_cp)
hcp = {'egyptian cat', 'tabby', 'tiger cat', 'radiator'}
hcp = list(hcp)

print({x: wn.synset(mappings[x]).min_depth() for x in vanilla_cp})
print({x: wn.synset(mappings[x]).min_depth() for x in hcp})

print([wn.synset(mappings[x]) for x in vanilla_cp])

vanilla_lca = get_lca([wn.synset(mappings[x]) for x in vanilla_cp])
print(vanilla_lca.lemmas(), vanilla_lca.min_depth())
hcp_lca = get_lca([wn.synset(mappings[x]) for x in hcp])
print(hcp_lca.lemmas(), hcp_lca.min_depth())

for x in vanilla_cp:
    hypernyms = wn.taxonomy.hypernym_paths(wn.synset(mappings[x]))[0]
    print(f"{x}: {[y.lemmas()[0] for y in hypernyms]}")

vanilla_proba(id_sample=3020, ps=vanilla_cp)

with open("data/imagenet/hcp_imagenet.pkl", 'rb') as openfile:
    hcp_scores = pickle.load(openfile)
scores = hcp_scores[3020]
id2label = {v.replace("omw-", "omw-subset-"): k for k, v in mappings.items()}
print(id2label)
print([mappings[x].replace("omw-", "omw-subset-") for x in hcp])
print({id2label[k.id]: float(v) for k, v in scores.items() if k.id in [mappings[x].replace("omw-", "omw-subset-") for x in hcp]})
# print({id2label[k.id]: v for k, v in scores.items() if k.id in [mappings[x].replace("subset-", "") for x in hcp]})