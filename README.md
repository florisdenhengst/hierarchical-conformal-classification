# Hierarchical Conformal Classification (HCC)

Code for the NeurIPS 2026 paper *Hierarchical Conformal Classification*. HCC produces conformal prediction
sets that may contain both leaf classes and internal nodes of a class hierarchy (a tree or a DAG), trading
off set size against specificity while keeping a finite-sample coverage guarantee.

- Poster: [NeurIPS 2026](https://neurips.cc/virtual/2026/loc/paris/poster/150172)
- Preprint: [ArXiv](https://arxiv.org/abs/2508.13288)
- Data (classifier scores, splits and taxonomies): [HuggingFace](https://huggingface.co/datasets/florisdh/hierarchical-conformal-classification)
- Webpage: [link](https://florisdh.nl/hierarchical-conformal-classification)

## Setup
Requires Python 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e MAPIE          # fork of MAPIE with the Westfall-Young maxT correction
python -m wn download omw-en:1.4
python -c "import wn; [wn.add(f) for f in ['models/dbpedia14/dbpedia14_lexicon.xml', 'models/gtzan/gtzan_lexicon.xml', 'models/imagenet/lexicon.xml']]"
```

The `dbp` and `gtz` score files are included in `models/`. Download the ImageNet scores from the Hugging Face
dataset and place them at `data/imagenet/predictions_val.csv`.

## Running
```bash
python hcc.py DATASET TAXONOMY MAPPING_FILE OUTPUT_FILE [TREE] [PER_LAYER] [ALPHA] [BETA] [SEED] \
              [LEAFS_ONLY] [METHOD] [DYNAMIC] [LOSS_APPROACH] [CP_METHOD] [--propagation leaf-cover]
```

| Argument | Values | Meaning |
|---|---|---|
| `TREE` | `True`/`False` | treat the hierarchy as a tree (one true node per cover) or a DAG |
| `PER_LAYER` | `True`/`False` | search one cover per layer (needed for ImageNet) or all NOL-covers |
| `ALPHA` | float | target error level |
| `BETA` | float or `betas` | trade-off between set size and covered leaves; `betas` sweeps 20 values |
| `LEAFS_ONLY` | `True`/`False` | standard CP on the leaves only (baseline) |
| `METHOD` | `hierarchical`, `multi-label` | HCC score, or the CRC recall-loss ablation |
| `DYNAMIC` | `True`, `False`, `nobc` | dynamic pruning; no dynamic pruning; no multiple-testing correction |
| `LOSS_APPROACH` | `first` (default in the paper), `last`, `costs` | which true node defines the nonconformity score |
| `CP_METHOD` | `lac`, `raps` | base conformal score |
| `--propagation` | `levels` (default), `leaf-cover` (used in the paper) | how classifier scores are propagated to internal nodes |

`SEED` seeds NumPy and Python; the calibration/test split itself is fixed (stratified 20/80, `random_state=42`).
Set `HCC_N_JOBS` to control parallelism. Each run writes a results CSV and a `*_timing.csv`.

| Script | Reproduces |
|---|---|
| `run_all.sh` | Table 1 (main results and ablations) |
| `run_all_betas.sh` | Figure: set size and covered leaves for varying β |
| `run_alphas.sh` | Figure: coverage, set size and covered leaves for varying α |
| `time_all.sh` | runtimes (runs with `sudo` and `nice`) |

Results are evaluated in `analyse_hcc.ipynb`. A full `run_all.sh` takes several hours, dominated by `dbp`
(about 50 minutes per HCC run); ImageNet runs need about 9 GB of RAM.

**Caching.** Inference and the default score propagation are cached in `/tmp` and `/tmp/cache`. Cache keys do
not include the code version; clear them after changing the code:
`rm -rf /tmp/cache /tmp/infer_mm_dynamic_*.pkl`.

## Tests
`MAPIE/mapie/tests/test_multiple_testing.py` tests the multiple-testing correction (`pytest`). Note that
`tests.py` removes and re-adds the `wn` lexicons in your local `wn` database.

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

## Attribution
Please cite

```
@inproceedings{denhengst2026hierarchical,
  title     = {Hierarchical Conformal Classification},
  author    = {den Hengst, Floris and Blin, In{\`e}s and Mohammadi, Majid and
               Shah, Syed Ihtesham and Younesian, Taranah},
  booktitle = {Advances in Neural Information Processing Systems},
  publisher = {Curran Associates, Inc.},
  volume = {39, Main Conference},
  year = {2026}
}
```
