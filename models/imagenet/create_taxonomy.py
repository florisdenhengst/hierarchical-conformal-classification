# -*- coding: utf-8 -*-
"""
The omw-en:1.4 taxonomy contains 117,659 synsets, however the classification task 
only uses a small subset of that taxonomy.

We create a taxonomy that is a subset of omw-en:1.4.

To do so, we retrieve that 1k images from the classification task that forms a list of synsets.
For each of these synsets, we retrieve its ancestor and descendants, and add them, 
including the synset, into the new taxonomy.
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import functools
import json
import click
import wn
from tqdm import tqdm

@functools.cache
def ancestors(node: wn.Synset) -> set[wn.Synset]:
    """ Get all ancestors of a node """
    path = set()
    pending = [node]
    while pending:
        synset = pending.pop()
        if synset != node:
            path.add(synset)
        parents = synset.hypernyms()
        if parents:
            pending.extend(parents)
    return path

@functools.cache
def descendants(node: wn.Synset) -> set[wn.Synset]:
    """ Get all descendants of a node """
    children = set()
    pending = [node]
    while pending:
        synset = pending.pop()
        children.add(synset)
        pending.extend(synset.hyponyms())
    return children


def get_subset_node_from_taxonomy(synsets: list[wn.Synset]) -> set[wn.Synset]:
    """ Return synsets {s1, ..., sn}, s.t. for each synset si, 
    all its ancestors and descendants are also in the set """
    output = set()
    to_explore = synsets
    while to_explore:
        synset = to_explore.pop()
        parents = ancestors(node=synset)
        # children = descendants(node=synset)
        output = output.union({synset}).union(parents)#.union(children)
        to_explore = list(set(to_explore).difference(output))#.difference(children))
    return output


def get_subset(ewn, file):
    """ From results get subset of nodes from the taxonomy """
    with open(file, 'r', encoding='utf-8') as openfile:
        info = json.load(openfile)
    nodes = list(info.keys())
    nodes = [ewn.synset(x) for x in nodes]
    subset = get_subset_node_from_taxonomy(synsets=nodes)
    return subset


def create_wn_xml(synsets, lexicon_id, version, output_file):
    """ Generate the WordNet lexicon from a list of synsets as XML 
    NB: follows the structure of previsouly created models, cf.
    "dbpedia14" and "gtzan" """
    root = ET.Element("LexicalResource", attrib={
        "xmlns:dc": "https://globalwordnet.github.io/schemas/dc/"
    })

    # Create the Lexicon element as a child of LexicalResource
    lexicon = ET.SubElement(root, "Lexicon", attrib={
        "id": lexicon_id,
        "label": f"Subset of omw-en",
        "language": "en",
        "version": version,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "email": "i.blin@vu.nl"
    })

    # Add synsets to the Lexicon
    for synset in tqdm(synsets):
        # Create the Synset element
        synset_element = ET.SubElement(lexicon, "Synset", attrib={
            # NOTE FdH: manually replace all IDs with unique IDs to avoid unexpected `wn` behaviors
            "id": synset.id,
            "ili": synset.ili.id if synset.ili else "",
            "partOfSpeech": synset.pos
        })

        # Add definitions
        definition_element = ET.SubElement(synset_element, "Definition")
        definition_element.text = synset.definition()

        # Add lemmas
        # for i, lemma in enumerate(synset.lemmas()):
        #     lexical_entry = ET.SubElement(lexicon, "LexicalEntry", attrib={
        #         # "id": f"lemma-{lemma.replace(' ', '-')}-{synset.id}",
        #         "id": f"lemma-{lemma}-{synset.pos}-{i}"
        #         })
        #     ET.SubElement(lexical_entry, "Lemma", attrib={
        #         "writtenForm": lemma,
        #         "partOfSpeech": synset.pos})
        #     ET.SubElement(lexical_entry, "Sense", attrib={
        #         "id": f"sense-{lemma}-{synset.id}-{i}",
        #         "synset": synset.id,})

        # Add relations (e.g., hypernyms, hyponyms)
        for relation_type, targets in synset.relations().items():
            for target in targets:
                if target.id in {synset.id for synset in synsets}:
                    ET.SubElement(synset_element, "SynsetRelation", attrib={
                        "target": target.id,
                        "relType": relation_type
                    })

    # Generate pretty XML string
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    # Add the XML declaration and DOCTYPE
    xml_with_doctype = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE LexicalResource SYSTEM "http://globalwordnet.github.io/schemas/WN-LMF-1.1.dtd">\n'
        + xml_str.split("\n", 1)[1]  # Skip the original XML declaration
    )

    # Save to a file
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(xml_with_doctype)

    print(f"Lexicon saved to {output_file}")


@click.command()
@click.argument("id_to_label")
@click.argument("lexicon_id")
@click.argument("version")
@click.argument("output_file")
def main(id_to_label, lexicon_id, version, output_file):
    """ Main """
    ewn = wn.Wordnet("omw-en:1.4")
    subset = set()
    subset = get_subset(ewn=ewn, file=id_to_label)
    print(f"Original # of synsets: {len(ewn.synsets())} | Filtered #: {len(subset)}")
    create_wn_xml(subset, lexicon_id, version, output_file)



if __name__ == '__main__':
    # python models/omw_subset/create_taxonomy.py data/imagenet/id_to_label.json omw-subset-en 1.4 models/omw_subset/lexicon.xml
    main()
