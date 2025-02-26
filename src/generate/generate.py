import argparse
import os
import tomli

import re

import regex

import pandas as pd

from src.generate.llama import LLaMaGenerate
from src.generate.input_data import InputData

def remove_prompt(row):
    return row["output_cleaned"][row["prompt_len"]:]

def fuzzy_find(row):
    disease_to_find = row["string"].lower()
    output = row["output_cleaned"].lower()

    found = regex.search(f"({re.escape(disease_to_find)}){{s<4}}", output)

    if found is not None:

        found_regex = found.span()

        return output[:found_regex[0]] + " <1CUI> " + output[found_regex[0]:found_regex[1]] + " </1CUI> " + output[found_regex[1]:]
    else:
        return None

def main(args):

    config_path = args.config
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    generation_config = config['generate']

    output_folder = generation_config['output_folder']

    input_file = generation_config['input_file']

    model_dir = generation_config['model_dir']

    generation_params = generation_config['generation_params']

    decoder_name = generation_params["model_dir"].replace("/", "-")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_file):
        input_data = InputData(input_file)

        input_data.generate()

    output_file_name = os.path.join(output_folder, f"generation_data_{decoder_name}.csv")

    post_processed_output_file_name = os.path.join(output_folder, f"generation_data_{decoder_name}_post_processed.csv")

    llama_umls_data = LLaMaGenerate(input_file, output_file_name, model_dir, generation_params)

    llama_umls_data.generate()

    df = pd.read_csv(output_file_name)

    df["prompt_len"] = df["prompt"].str.len()

    df["output_cleaned"] = df["output"].str.replace("</s>", "").str.replace("<s>", "").str[1:]

    df["output_cleaned"] = df.apply(remove_prompt, axis=1)

    df["matched_output"] = df.apply(fuzzy_find, axis=1)

    df  = df[~df["matched_output"].isna()]

    df[["cui", "matched_output"]].to_csv(post_processed_output_file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)
