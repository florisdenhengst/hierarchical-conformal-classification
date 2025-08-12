#!/bin/bash
# runs in parallel with &
# replace & by && to run sequentially
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_betas_first.csv False False 0.1 betas 42 False hierarchical True first lac
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_betas_first.csv True False .0003 betas 42 False hierarchical True first lac
python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_betas_first.csv False True 0.02 betas 42 False hierarchical True first lac
echo "done"
