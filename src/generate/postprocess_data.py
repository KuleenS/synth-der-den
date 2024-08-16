import argparse

import regex
import re
import pandas as pd

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


def main(input, output):
    df = pd.read_csv(input)

    df["prompt_len"] = df["prompt"].str.len()

    df["output_cleaned"] = df["output"].str.replace("</s>", "").str.replace("<s>", "").str[1:]

    df["output_cleaned"] = df.apply(remove_prompt, axis=1)

    df["matched_output"] = df.apply(fuzzy_find, axis=1)

    df  = df[~df["matched_output"].isna()]

    df[["cui", "matched_output"]].to_csv(output, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')

    args = parser.parse_args()

    main(args.input, args.output)