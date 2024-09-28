import argparse

import csv

import os

import datasets

import evaluate

import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForTokenClassification

from tqdm.auto import tqdm

from src.ner.file_processor import FileProcessor
from src.ner.custom_collator import DataCollatorForTokenClassificationCustom
from src.ner.utils import NERDataset

def main(args):

    file = args.file
    files = args.files

    file_list = None

    conll_output = None

    if file is None:
        file_list = files
        conll_output = [os.path.join(args.conll_output, x) for x in file_list]
    elif files is None:
        file_list = [file]
        conll_output = [args.conll_output]
    else:
        raise ValueError("Must specify file or files")

    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    label_names = ["O", "DISEASE"]

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    for conll_file, output_file in tqdm(file_list, conll_output):

        file_processor = FileProcessor()

        test_sentences, test_labels = file_processor.process_file(conll_file, test=True)

        test_labels = [[label2id[y] for y in x if y in label2id] for x in test_labels]

        test_dataset = NERDataset(tokenizer, test_sentences, test_labels)

        data_collator = DataCollatorForTokenClassificationCustom(tokenizer)

        test_dataloader = DataLoader(
            test_dataset, collate_fn=data_collator, batch_size=32
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)
        
        model.eval()

        total_predictions = []
        total_tokens = []

        for batch in tqdm(test_dataloader):

            word_ids = batch["word_ids"]

            del batch["word_ids"]

            batch.to(device)

            with torch.no_grad():

                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)

            tokens = batch["input_ids"]
            
            for i in range(len(tokens)):

                sentence = tokens[i].detach().cpu().tolist()

                sentence_word_ids = word_ids[i].detach().cpu().tolist()

                sentence_predictions = predictions[i]

                num_words = max(sentence_word_ids) + 1

                words = [""] * num_words

                word_level_sentence_predictions = []

                for token,word_id, token_prediction in zip(sentence, sentence_word_ids, sentence_predictions):

                    if word_id != -100:

                        token_string = tokenizer.convert_ids_to_tokens(token)

                        if len(words[word_id]) == 0:
                            word_level_sentence_predictions.append(token_prediction.item())
                        
                        words[word_id] += token_string
                
                words = [x.replace("##", "") for x in words]

                total_tokens.extend(words)

                total_predictions.extend(word_level_sentence_predictions)

                total_tokens.extend(" ")

                total_predictions.extend(" ")

        total_predictions = [id2label[x] if x != " " else " " for x in total_predictions ]

        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=" ", quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerows(zip(total_tokens, total_predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('-f','--file', required=False)
    parser.add_argument('-f','--files', nargs="+",required=False)
    parser.add_argument('-m','--model_checkpoint')
    parser.add_argument('-c','--conll_output')

    args = parser.parse_args()
    
    main(args)
