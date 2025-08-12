# -*- coding: utf-8 -*-
"""
Converting hcp output to json, to be able to run user_study/imagenet/get_data.py
"""
import re
import json
import click
import pandas as pd

def extract_synsets(s):
    """Extracts the elements inside Synset('...') from a string and returns them as a list."""
    return re.findall(r"Synset\('([^']+)'\)", s)


@click.command()
@click.argument("hcp_csv")
@click.argument("hcp_json")
def main(hcp_csv, hcp_json):
    """ Main """
    df = pd.read_csv(hcp_csv, index_col=0)
    res = {}
    for index, row in df.iterrows():
        res[index] = {
            "label": extract_synsets(s=row.label)[0],
            "set": extract_synsets(s=row.prediction_set)
        }
    with open(hcp_json, 'w', encoding='utf-8') as openfile:
        json.dump(res, openfile, indent=4)


if __name__ == '__main__':
    # python process_data/imagenet/convert_hcp_to_json.py data/imagenet/0.05/hcp_imagenet.csv data/imagenet/0.05/results10k_omw_hcp.json 
    main()