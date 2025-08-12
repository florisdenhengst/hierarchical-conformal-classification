# hierarchical-conformal-prediction

## Prerequisites
Python 3.10.14

## Installation and setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You need to install and add the lexicons we used in our experiments:
```python
import wn
wn.add("models/dbpedia14/dbpedia14_lexicon.xml")
wn.add("models/gtzan/gtzan_lexicon.xml")
wn.add("./models/imagenet/lexicon.xml")
```

```bash
python -m wn download omw-en:1.4
```

Download the imagenet files [here](https://drive.google.com/file/d/1woeukWQTV0YAvKWPpRCpE_GiefN-nuPU/view?usp=sharing), place the ``.tar.gz`` file in ``data/imagenet/`` and uncompress.

## Code Structure
The code is structured as follows:
* `hcc.py` main entrypoint for hierarchical conformal classification
* `run_all.sh` and `run_all_betas.sh` are scripts to create the main results in the paper. 
* `algo` contains all the main algorithm and helpers to perform the hierarchical conformal prediction.
* `data` contains the various data used for our three taxonomies. Each taxonomy has a dedicated folder. Below are usual indicators on the files' names:
    * Contains `id2label` or `id_to_label`: mapping `wn` taxonomy id to human-readable label;
    * Contains `results`: results of the vanilla conformal predictors. If the title additionally contains `hcp`, this is results for our hierarchical conformal predictor, and was derived from the `hcp_{dataset}.csv` file;
    * `hcp_{dataset}.csv`: output of the `hcp.py` script;
    * Contains `label2synset`: reverse of `id2label`.
* `models` contains all scripts related mostly to (i) creating the taxonomy (ii) (training) the base classifiers. There is also a test lexicon that is used for examples.
* `user_study` contains the scripts and samples related to our user study, as well as the analysis of the results
* `examples` contains various function tests and example scripts for the hierarchical conformal prediction.
* `process_data` contains various files, taxonomy-specific, to convert the data into correct formats.
    * `dbpedia14`
        * `convert_dbpedia.py`: convert output from vanilla CP to another format
    * `imagenet`
        * `convert_hcp_to_json.py`: convert results from `hcp.py` to a `.json` file, comparable to the output of vanilla conformal predictors;
        * `convert_imagenet_to_label.py`: convert internet file to `data/imagenet/mappings.json`;
        * `convert_results_nb_to_omw.py`: convert the raw output of RAPS to a format that can be directly used by the `wn` taxonomy;
        * `get_proba_file_in.py`: from the `scores.npy` and the `results` file, get a `.csv` to be used as input for hierarchical conformal prediction;
        * `id_to_label.py`: mappings to json
        * `transform_imagenet_mapping.py`: from `.txt` to `.json`




## Additional references
* [wn documentation](https://wn.readthedocs.io/en/latest/)
* [mappings imagenet2wordnet](https://gist.github.com/fnielsen/4a5c94eaa6dcdf29b7a62d886f540372)

