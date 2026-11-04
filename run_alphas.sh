#!/bin/bash

for alpha in 0.001 0.01 0.02 0.05 .1 .15 .2 .3; do
    echo "${alpha}"
    python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_${alpha}_0.19.csv True False $alpha 0.19 42 False hierarchical True first lac --propagation leaf-cover &&
    python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_${alpha}_0.04.csv False True $alpha 0.04 42 False hierarchical True first --propagation leaf-cover &&
    python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_${alpha}_0.25.csv False True $alpha .25 42 False hierarchical True first lac --propagation leaf-cover &&
    echo "done {$alpha}"
done

# data/imagenet/hcp_0.02_0.04_first.csv False True 0.02 0.04 42 False hierarchical True first
for alpha in 0.001 0.01 0.02 0.05 .1 .15 .2 .3; do
    echo "${alpha}"
    python hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_${alpha}_leafs.csv False True $alpha 0.04 42 True --propagation leaf-cover
    python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_${alpha}_leafs.csv True True $alpha .19 42 True --propagation leaf-cover
    python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_${alpha}_leafs.csv True True $alpha .25 42 True --propagation leaf-cover
done
