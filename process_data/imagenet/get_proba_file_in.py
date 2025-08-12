# -*- coding: utf-8 -*-
"""
Get probability file for Imagenet
"""
import os
import ast
import json
import click
import numpy as np
import pandas as pd
from scipy.special import softmax

def get_label_to_id(file):
    """ Read mappings """
    with open(file, 'r', encoding='utf-8') as f:
        labels = f.read()
    st = ast.literal_eval(labels)
    res = {f"omw-en-{v['uri'].split('/')[-1]}": k for k, v in st.items()}
    return res

@click.command()
@click.argument("scores_npy")
@click.argument("label2synset")
@click.argument("result_omw")
@click.argument("folder_img")
@click.argument("save")
def main(scores_npy, label2synset, result_omw, folder_img, save):
    """ Main: combine all """
    scores = np.load(scores_npy)
    probabilities = softmax(scores, axis=-1)

    label_to_id = get_label_to_id(file=label2synset)
    with open(result_omw, 'r', encoding='utf-8') as openfile:
        results = json.load(openfile)

    data = [(i, os.path.join(folder_img, f"{str(i)}.png"), label_to_id[info["label"]]) \
        for i, info in results.items()]
    df = pd.DataFrame(data)
    df = pd.concat([df, pd.DataFrame(probabilities)], axis=1)
    df.columns = ["id", "content", "label"] + \
        [f"label_{str(i)}" for i in range(probabilities.shape[1])]
    df.to_csv(save, index=False)

    return df



if __name__ == '__main__':
    # python process_data/imagenet/get_proba_file_in.py ./data/imagenet/0.05/scores.npy ./data/imagenet/imagenet_label_to_wordnet_synset.txt ./data/imagenet/0.05/results10k_omw.json ./data/imagenet/0.05/images10k ./data/imagenet/0.05/probas.csv
    main()
