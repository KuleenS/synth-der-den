import argparse

import csv

import os

import torch

from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForTokenClassification

from tqdm.auto import tqdm

def main(args):

    file = args.file
    files = args.files

    file_list = None

    conll_output = None

    if file is None:
        file_list = files
        conll_output = [os.path.join(args.conll_output, os.path.basename(x).replace(".txt", ".conll")) for x in file_list]
    elif files is None:
        file_list = [file]
        conll_output = [args.conll_output.replace(".txt", ".conll")]
    else:
        raise ValueError("Must specify file or files")

    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

    tokenizer.model_max_length = 510

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device, batch_size=4, aggregation_strategy="average", stride=0)

    for conll_file, output_file in tqdm(zip(file_list, conll_output)):
        with open(conll_file, "r") as f:
            text = f.read()

        result = ner_pipeline(text)

        tokens = []
        labels = []

        for entity in result:

            split_words = entity["word"].split(" ")

            tokens.extend(split_words)
            
            if entity["entity_group"] == "No Disease":
                labels.extend(len(split_words) * ["O"])
            else:
                labels.extend(len(split_words) * ["DISEASE"])
        
        subword_removed_tokens = []
        subword_removed_labels = []

        for token,label in zip(tokens, labels):
            if token.startswith('##'):
                token = subword_removed_tokens[len(subword_removed_tokens)-1] + token.replace('##','')

                if subword_removed_labels[len(subword_removed_labels)-1] == "DISEASE" or label == "DISEASE":
                    label = "DISEASE"
                else:
                    label = "O"

                subword_removed_tokens.pop()
                subword_removed_labels.pop()

            subword_removed_tokens.append(token)
            subword_removed_labels.append(label)
        
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=" ", quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerows(zip(subword_removed_tokens, subword_removed_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=False)
    parser.add_argument('--files', nargs="+",required=False)
    parser.add_argument('-m','--model_checkpoint')
    parser.add_argument('-c','--conll_output')

    args = parser.parse_args()
    
    main(args)
