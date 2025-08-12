# -*- coding: utf-8 -*-
"""
Rename ImageNet for the user study
"""
import os
import subprocess
from tqdm import tqdm
import pandas as pd

F_OUT = "aaai/imagenet_sorted"

def load_data(data_path: str = "aaai/in_results.tsv") -> pd.DataFrame:
    """
    Load the data from the CSV file.
    """
    df = pd.read_csv(data_path, sep="\t")[["index", "Slide #"]].rename(columns={"Slide #": "slide"})
    return df


def main():
    df = load_data()
    if not os.path.exists(F_OUT):
        os.makedirs(F_OUT)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        old_name = f"{row['index']}.JPEG"
        new_name = f"{row['slide']}.JPEG"
        cmd = f"cp aaai/imagenet/{old_name} {F_OUT}/{new_name}"
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()