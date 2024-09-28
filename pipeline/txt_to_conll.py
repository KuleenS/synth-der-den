import argparse

import os

import spacy

def main(args):
    

    list_of_text_filse = [os.path.join(args.input_path, x) for x in os.listdir(args.input_path)]

    nlp = spacy.load("en_core_sci_md")

    for filename in list_of_text_filse:
        tokens = []
        labels = []

        with open(filename, "r") as f:
            text_input = f.read()

        #loads pipefiles
        doc = nlp(text_input)

        #remove overlaps and have clean offsets for labelling of the spacy doct
        #get the labelled tokens from spacy doc  
        for sent in doc.sents:

            tokens_sent = [x.text for x in sent]
            iob_tags = ["O" for t in sent]

            tokens.extend(tokens_sent)
            labels.extend(iob_tags)

            tokens.extend(' ')
            labels.extend(' ')

        with open(os.path.join(args.output_folder, os.path.basename(filename)[:-4]+".conll"), "w"):
            for token, label in zip(tokens, labels):
                f.write(f'{token} {label}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_folder')
    parser.add_argument('-o','--output_folder')

    args = parser.parse_args()
    
    main(args)
