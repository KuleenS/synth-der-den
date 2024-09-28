import argparse

import codecs

import os

import spacy

def main(args):
    list_of_text_files = [os.path.join(args.input_folder, x) for x in os.listdir(args.input_folder)]

    nlp = spacy.load("en_core_sci_md")

    for filename in list_of_text_files:
        print(filename)
        tokens = []
        labels = []

        with codecs.open(filename, "r", encoding='utf-8', errors="ignore") as f:
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
        
        with open(os.path.join(args.output_folder, os.path.basename(filename)[:-4]+".conll"), "w") as f:
            for token, label in zip(tokens, labels):
                f.write(f'{token} {label}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_folder')
    parser.add_argument('-o','--output_folder')

    args = parser.parse_args()
    
    main(args)
