#!/bin/bash
# runs in parallel with &
# replace & by && to run sequentially
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.25_first_nd.csv False False 0.1 .25 42 False hierarchical False first lac 
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.25_first.csv False False 0.1 .25 42 False hierarchical True first lac
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.25_first_layer.csv False True 0.1 .25 42 False hierarchical True first lac
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.25_nobc.csv False False 0.1 .25 42 False hierarchical nobc first lac
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.25_ml.csv False False 0.1 .25 42 False multi-label
python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_leafs.csv True True 0.1 .25 42 False
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_0.19_first_nd.csv True True .0003 .19 42 False hierarchical False first lac
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_0.19_first.csv True False .0003 .19 42 False hierarchical True first lac
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_0.19_first_layer.csv True True .0003 .19 42 False hierarchical True first lac
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_0.19_nobc.csv True True .0003 .19 42 False hierarchical nobc first lac
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_0.19_ml.csv True True .0003 .19 42 False multi-label
python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_leafs.csv True True .0003 .19 42 True
python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_0.04_first_nd.csv False True 0.02 0.04 42 False hierarchical False first lac
python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_0.04_first.csv False True 0.02 0.04 42 False hierarchical True first
python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_0.04_nobc.csv False True 0.02 0.04 42 False hierarchical nobc first
python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_0.04_ml.csv False True 0.02 0.04 42 False multi-label
python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_leafs.csv False True 0.02 0.04 42 True
echo "done"
