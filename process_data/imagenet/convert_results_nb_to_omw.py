
import ast
import json
import click

@click.command()
@click.argument("input_f")
@click.argument("output_f")
def main(input_f, output_f):

    with open('./data/imagenet/imagenet_label_to_wordnet_synset.txt', 'r', encoding='utf-8') as f:
        labels = f.read()
        st = ast.literal_eval(labels)
        labeldict = {key: value['label'].split(',')[0] for key, value in st.items()}
    
    with open('data/imagenet/imagenet_mapping.json', 'r') as openfile:
        omw_to_label = json.load(openfile)
    label_to_omw = {y: f"omw-en-{key[1:]}-n" for key, val in omw_to_label.items() for y in val.split(", ")}

    with open(input_f, 'r', encoding='utf-8') as openfile:
        input_data = json.load(openfile)
    
    output = {}
    for key, val in input_data.items():
        set_ = val["set"] if not isinstance(val["set"][0], list) else val["set"][0]
        output[key] = {
            "label": label_to_omw[labeldict[val["label"]]],
            "set": [label_to_omw[labeldict[x]] for x in set_]
        }
    
    with open(output_f, "w", encoding="utf-8") as openfile:
        json.dump(output, openfile, indent=4)


if __name__ == '__main__':
    # python process_data/imagenet/convert_results_nb_to_omw.py data/imagenet/0.05/results10k_number.json data/imagenet/0.05/results10k_omw.json
    main()

