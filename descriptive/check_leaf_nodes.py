# -*- coding: utf-8 -*-
"""
Checking whether prediction sets contain leaf nodes only
"""
import json
from tqdm import tqdm
from cv_ps_metrics import get_synsets

def check_results(json_file):
    """ Check prediction sets from results """
    with open(json_file, encoding='utf-8') as openfile:
        results = json.load(openfile)
    ps = [x["set"] for _, x in results.items()]
    ps = [get_synsets(x) for x in ps]
    ln_different_ps = 0
    for curr_ps in tqdm(ps):
        if any(x.hyponyms() for x in curr_ps):
            ln_different_ps += 1
    print(f"{len(ps)} prediction sets, {ln_different_ps} where leaf nodes are different than the prediction sets")
    


if __name__ == '__main__':
    for file in [
        "./data/dbpedia14/dbpedia14_results_0.0007_omw.json",
        "./data/gtzan/gtzan_results_0.2.json",
        "./data/imagenet/0.05/results10k_omw.json"
    ]:
        print(file)
        check_results(json_file=file)
        print("===")
