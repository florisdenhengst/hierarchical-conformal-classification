#!/bin/bash

# Execute the whole script with sudo: sudo ./run_benchmarks.sh
# Runs sequentially using '&&'. If a command fails, execution stops.

RUN_CMD="sudo caffeinate -u nice -n -20 /usr/bin/time -p python"

# echo "=== Running GTZAN First ===" && \
# $RUN_CMD hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.25_first.csv False False 0.1 .25 42 False hierarchical True first lac --propagation leaf-cover && \

# echo "=== Running GTZAN Leafs ===" && \
# $RUN_CMD hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_leafs.csv True True 0.1 .25 42 False --propagation leaf-cover && \

# echo "=== Running DBPedia14 First ===" && \
# $RUN_CMD hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_0.19_first.csv True False .0003 .19 42 False hierarchical True first lac --propagation leaf-cover && \

# echo "=== Running DBPedia14 Leafs ===" && \
# $RUN_CMD hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.0003_leafs.csv True True .0003 .19 42 True --propagation leaf-cover && \

# echo "=== Running ImageNet First ===" && \
$RUN_CMD hcc.py data/imagenet/predictions_val_sample.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_0.04_first.csv False True 0.02 0.04 42 False hierarchical True first --propagation leaf-cover && \

# echo "=== Running ImageNet Leafs ===" && \
# $RUN_CMD hcc.py data/imagenet/predictions_val.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.02_leafs.csv False True 0.02 0.04 42 True --propagation leaf-cover && \

echo "done"
