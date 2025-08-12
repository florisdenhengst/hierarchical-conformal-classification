# -*- coding: utf-8 -*-
"""
Convert dbpedia results (current ./data/dbpedia14/dbpedia_results_0.0001.json) into a format that "cv_ps_metrics.py" can process
"""
import json

INPUT_F = "./data/dbpedia14/dbpedia_results_0.005.json"
OUTPUT_F = "./data/dbpedia14/dbpedia_results_0.005_omw.json"

with open(INPUT_F, encoding='utf-8') as openfile:
    data = json.load(openfile)

for _, info in data.items():
    info["set"] = [f"dbpedia14-en-{x}-synset-0-n" for x in info["set"]]

with open(OUTPUT_F, 'w', encoding='utf-8') as openfile:
    json.dump(data, openfile, indent=4)
