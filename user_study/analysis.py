# -*- coding: utf-8 -*-
"""
Analyze results of the user study
"""
import numpy as np
import pandas as pd
from collections import Counter
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import binomtest

FP = {
    k: f"aaai_user_study/user_study/userstudy_sample_{k}.xlsx" for k in ["dbpedia", "imagenet"]
}
COL_ANNOT = "num_annotations"
LABELS = ["no preference", "vanilla", "hcp"]
TEMPLATE_P = "aaai/template.txt"
NB_SAMPLE = 250

def majority_vote(row):
    """ majority vote """
    votes = row.dropna()
    if len(votes) == 0:
        return np.nan  # No annotations
    return votes.mode().values[0] if len(votes.mode()) == 1 else "tie"


def read_data(file_path: str) -> dict:
    """
    Read data from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file.
        
    Returns:
        dict: Dictionary of DataFrames, one for each sheet.
    """
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    base_df = None
    annot_dfs = []
    for sheet_num, (_, df) in enumerate(all_sheets.items(), 1):
        # Take only the first NB_SAMPLE rows
        df_subset = df.head(NB_SAMPLE).copy()
        
        # Rename columns
        df_subset = df_subset.rename(columns={
            'annotation': f'annot_{sheet_num}',
            'ground truth': 'ground_truth',
            'prediction_set_hcp': 'hcp',
            'prediction_set_vanilla': 'vanilla'
        })
        
        if base_df is None:
            # First sheet: keep the base columns + first annotation
            base_df = df_subset[['ground_truth', 'hcp', 'vanilla', f'annot_{sheet_num}']]
        else:
            # Subsequent sheets: only keep the annotation column
            annot_df = df_subset[[f'annot_{sheet_num}']]
            annot_dfs.append(annot_df)
    # Concatenate all annotation columns to the base dataframe
    if annot_dfs:
        result = pd.concat([base_df] + annot_dfs, axis=1)
    else:
        result = base_df
    return result


def describe_data(data: dict) -> None:
    """
    Print the description of the data.
    
    Args:
        data (dict): Dictionary of DataFrames.
    """
    cols_annot = [col for col in data.columns if col.startswith('annot_')]
    print(f"Number of annotators: {len(cols_annot)}")
    print(f"# annotated samples: {[len(data[~data[col].isna()]) for col in cols_annot]}")


def prep_df(df):
    """ self explanatory """
     # remove annotator column with no values
    df = df.dropna(axis=1, how='all').copy()
    annot_cols = [x for x in df.columns if x.startswith("annot_")]
    df[COL_ANNOT] = df[annot_cols].notna().sum(axis=1)
    for l in LABELS:
        df[f"count_{l.replace(' ', '_')}"] = (df[annot_cols] == l).sum(axis=1)
    for col in ["vanilla", "hcp"]:
        df[col] = df[col].apply(lambda x: set(x.split(", ")))
    df["same_cp"] = df["vanilla"] == df["hcp"]
    for x in LABELS:
        df[f"prop_{x.replace(' ', '_')}"] = df[f"count_{x.replace(' ', '_')}"] / df[COL_ANNOT]
    df["majority_preference"] = df[annot_cols].apply(majority_vote, axis=1)
    df["gt_in_vanilla"] = df.apply(lambda row: row["ground_truth"] in row["vanilla"], axis=1)
    df["gt_in_hcp"] = df.apply(lambda row: row["ground_truth"] in row["hcp"], axis=1)

    return df, annot_cols


def print_info(tp, **info):
    if tp == "same_ps":
        print(f"""
        There were {len(info['annot_col'])} annotators. Out of {info['df'].shape[0]} data samples to annotate, {info['df'][info['df'].same_cp].shape[0]} had the same vanilla and hierarchical prediction sets.
        The rest of the analysis focuses on the {info['df'][~info['df'].same_cp].shape[0]} cases where the prediction sets were different. 
        """)


def generic_all(df):
    """ Generic - all results (whether ps is the same or not) """
    # Stats on num annotations
    num_annotations = Counter(df[COL_ANNOT].values)
    print(f"# of annotations distributions: {num_annotations}")
    print(f"The average number of annotations per data point was: {round(np.mean(df[COL_ANNOT].values), 2)}")

    print(df.groupby(["gt_in_vanilla", "gt_in_hcp"]).agg({"ground_truth": "count"}))


def genetric_metrics(df, annot_cols):
    """ Descriptive """
    # Flatten all annotations into a single list
    all_annotations = df[annot_cols].values.flatten()
    all_annotations = all_annotations[~pd.isna(all_annotations)]  # Remove NaNs

    # Count occurrences of each label
    preference_counts = Counter(all_annotations)
    print("Annotation distribution:", preference_counts)

    # Convert to percentages
    total_annotations = sum(preference_counts.values())
    preference_percentages = {k: v / total_annotations * 100 for k, v in preference_counts.items()}
    print("Percentage of each choice:", preference_percentages)

    # Majority votes
    majority_counts = Counter(df.majority_preference.values)
    print("Majority distribution:", majority_counts)

    # Majority votes - perc.
    total_maj = sum(majority_counts.values())
    majority_percentages = {k: v / total_maj * 100 for k, v in majority_counts.items()}
    print("Percentage of each choice:", majority_percentages)


def annotator_agreement(df):
    """ se """
    num_weighted_kappa = 0
    denom_weighted_kappa = 0
    for num_annot, group in df.groupby(COL_ANNOT):
        matrix = group[[f"count_{x.replace(' ', '_')}" for x in LABELS]].values
        kappa = fleiss_kappa(matrix)
        print(f"# annotators: {num_annot}, FK: {kappa}")
        num_weighted_kappa += kappa*group.shape[0]
        denom_weighted_kappa += group.shape[0]
    return_kappa = round(num_weighted_kappa/denom_weighted_kappa, 2)
    print("Weighted kappa: ", return_kappa)
    return return_kappa


def binom_test(df):
    """ Binomial test """
    total_annotators = df['count_vanilla'].sum() + df['count_hcp'].sum()
    hierarchical = df['count_hcp'].sum()

    # Run binomial test (expected proportion = 50%)
    result = binomtest(hierarchical, total_annotators, p=0.5, alternative='greater')

    # Print results
    print(f"P-value: {result}")

def populate_print_template(template_p, **kwargs):
    with open(template_p, 'r', encoding='utf-8') as f:
        template = f.read()
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    print(template)


def get_kappa_interpretation(kappa):
    if kappa < 0.2:
        return "poor"
    elif kappa < 0.4:
        return "fair"
    elif kappa < 0.6:
        return "moderate"
    elif kappa < 0.8:
        return "strong"
    else:
        return "very strong"

def get_kappa_comparison_template(kappa_db, kappa_in):
    inter_db = get_kappa_interpretation(kappa_db)
    inter_img = get_kappa_interpretation(kappa_in)
    if inter_db == inter_img:
         return f"This indicates a {inter_db} level of agreement across annotators for both datasets."
    else:
        small_inter, small_ds = (inter_db, "dbp") if kappa_db < kappa_in else (inter_img, "img")
        large_inter, large_ds = (inter_db, "db") if kappa_db > kappa_in else (inter_img, "img")
        return "".join([
            "This indicates a ", small_inter, " (\texttt{", small_ds,
            "}) to ", large_inter, " (\texttt{", large_ds, "}) level of agreement."])

def main():
    data_all = {k: read_data(v) for k, v in FP.items()}
    template_data = {}
    for dataset, data in data_all.items():
        print(f"Dataset: {dataset}")
        describe_data(data)
        df, annot_cols = prep_df(data)
        print_info("same_ps", df=df, annot_col=annot_cols)
        generic_all(df=df)
        genetric_metrics(df=df[~df.same_cp], annot_cols=annot_cols)
        kappa = annotator_agreement(df=df[~df.same_cp])
        template_data[f"kappa_{dataset}"] = kappa
        binom_test(df=df[~df.same_cp])
        print("========")

    template_data["kappa_comparison_sent"] = get_kappa_comparison_template(
        template_data["kappa_dbpedia"], template_data["kappa_imagenet"]
    )

if __name__ == "__main__":
    main()
