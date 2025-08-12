# -*- coding: utf-8 -*-
"""
Describe taxonomies used in our examples
"""
import json
from collections import Counter
import wn
from wn.taxonomy import min_depth
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def describe_tax(tax_name):
    """ describe taxonomy """
    ewn = wn.Wordnet(tax_name)
    synsets = ewn.synsets()
    depths = [min_depth(s, simulate_root=True) for s in synsets]
    leaves = wn.taxonomy.leaves(ewn, pos='n')
    depths_leaves = [min_depth(s, simulate_root=True) for s in leaves]

    return {
        "depth": max(depths),
        "nb_synsets": len(synsets),
        "nb_leaf_nodes": len(leaves),
        "synset_per_level": Counter(depths),
        "breadth": round(np.mean(list(Counter(depths).values())), 1),
        "leaves_per_level": Counter(depths_leaves)
    }

def describe_data(res_file):
    """ describe data """
    with open(res_file, 'r', encoding="utf-8") as openfile:
        res = json.load(openfile)
    
    return {
        "nb_data": len(res)
    }

def create_funnels(data_depth, data_leaves, save_path):
    """ funnel visualisation """
    blue_color, green_color = 'royalblue', 'forestgreen'
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=list(data_depth.keys()),  
        shared_yaxes=True, horizontal_spacing=0.02,
    )
    col = 1
    for k, count in data_depth.items():
        number_no = [count[x] - data_leaves[k].get(x, 0) for x in sorted(count.keys())]
        number_yes = [data_leaves[k].get(x, 0) for x in sorted(count.keys())]

        stage = [f"Level {str(x)}" for x in sorted(count.keys())]

        show_legend = True if col == 1 else False
        fig.add_trace(go.Funnel(
            y=stage, x=number_no, name="no",
            marker={"color": blue_color}, showlegend=show_legend),
            row=1, col=col)
        fig.add_trace(go.Funnel(
            y=stage, x=number_yes, name="yes",
            marker={"color": green_color}, showlegend=show_legend),
            row=1, col=col)

        col += 1

    fig.update_layout(
        title_text="Synsets across levels of the taxonomy",
        title_x=0.5, height=500, width=800,  
        showlegend=True
    )

    fig.write_image(save_path, format='pdf')

def main():
    """ Main, all lexicons """
    lexicons = [
        ('dbpedia14-en', 'models/dbpedia14/dbpedia14_lexicon.xml', './data/dbpedia14/dbpedia_results_0.005.json'),
        #('hcptest-en', 'models/test/test_lexicon.xml'),
        ('omw-subset-en', 'models/imagenet/lexicon.xml', "./data/imagenet/0.05/results10k_omw.json"),
        ('gtzan-en', 'models/gtzan/gtzan_lexicon.xml', './data/gtzan/gtzan_results_0.2.json'),
    ]
    data_funnel_depths = {}
    data_funnel_leaves = {}
    for lexicon, file, results_f in lexicons:
        try:
            wn.remove(lexicon)
        except wn.Error:  # the lexicon was never added
            pass
        wn.add(file)

        tax_stats = describe_tax(tax_name=lexicon)
        data_stats = describe_data(res_file=results_f)
        print(f"---{lexicon}\nTaxonomy stats: {tax_stats}\nData stats: {data_stats}")
        data_funnel_depths[lexicon.replace("omw-subset-en", "wordnet-subset")] = tax_stats["synset_per_level"]
        data_funnel_leaves[lexicon.replace("omw-subset-en", "wordnet-subset")] = tax_stats["leaves_per_level"]
    
    create_funnels(data_depth=data_funnel_depths, data_leaves=data_funnel_leaves, save_path="images/funnel_plot.pdf")


if __name__ == '__main__':
    main()