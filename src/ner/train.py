import argparse

import csv

import os

import datasets

import evaluate

import torch

from torch.utils.data import DataLoader

from torch.optim import AdamW

import pandas as pd

from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler

from tqdm.auto import tqdm

from src.ner.file_processor import FileProcessor
from src.ner.custom_collator import DataCollatorForTokenClassificationCustom
from src.ner.utils import postprocess, NERDataset, NERDatasetCUI

def main(args):

    files = args.files

    model_checkpoint = args.model_checkpoint

    output_dir = args.model_output

    conll_output = args.conll_output

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    label_names = ["O", "DISEASE"]

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    if len(files) == 3:
        file_processor = FileProcessor()

        train_sentences, train_labels = file_processor.process_file(files[0])
        valid_sentences, valid_labels = file_processor.process_file(files[1])
        test_sentences, test_labels, test_cuis = file_processor.process_test_cui_file(files[2])

        train_labels = [[label2id[y] for y in x if y in label2id] for x in train_labels]
        valid_labels = [[label2id[y] for y in x if y in label2id] for x in valid_labels]
        test_labels = [[label2id[y] for y in x if y in label2id] for x in test_labels]

        training_dataset = NERDataset(tokenizer, train_sentences, train_labels)
        valid_dataset = NERDataset(tokenizer, valid_sentences, valid_labels)

        all_test_cuis = sum(test_cuis, [])
        
        all_test_cuis = set(all_test_cuis)

        id2cui = {i+1: label for i, label in enumerate(all_test_cuis)}

        id2cui[0] = "NOCUI"

        cui2id = {v: k for k, v in id2cui.items()}

        test_cuis_mapped = [[cui2id[x] for x in y] for y in test_cuis]

        test_dataset = NERDatasetCUI(tokenizer, test_sentences, test_labels, test_cuis_mapped)

    data_collator = DataCollatorForTokenClassificationCustom(tokenizer)

    train_dataloader = DataLoader(
        training_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=64,
    )
    
    eval_dataloader = DataLoader(
        valid_dataset, collate_fn=data_collator, batch_size=32
    )

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

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = 10 * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    current_eval_loss = 10000000

    for epoch in range(10):
        # Training
        model.train()
        for batch in train_dataloader:

            word_ids = batch["word_ids"]

            del batch["word_ids"]

            batch.to(device)

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()

        metric = evaluate.load("seqeval", experiment_id=os.environ["PYTHONHASHSEED"])

        eval_loss = 0

        for batch in eval_dataloader:
            with torch.no_grad():

                word_ids = batch["word_ids"]

                del batch["word_ids"]

                batch.to(device)
                
                outputs = model(**batch)

                loss = outputs.loss

                eval_loss += loss.detach().cpu().item()

            predictions = outputs.logits.argmax(dim=-1).detach().cpu()
            labels = batch["labels"].detach().cpu()

            true_predictions, true_labels = postprocess(predictions, labels, label_names)

            metric.add_batch(predictions=true_predictions, references=true_labels)
        
        average_loss = eval_loss / len(eval_dataloader)

        results = metric.compute()

        print(
            f"epoch {epoch+1}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )

        print(f"Eval Loss: {average_loss}")

        if average_loss < current_eval_loss:
            current_eval_loss = average_loss
        else:
            break
    
    model.eval()

    total_predictions = []
    total_labels = []
    total_tokens = []
    total_cuis = []

    metric = evaluate.load("seqeval", experiment_id=os.environ["PYTHONHASHSEED"])

    for batch in tqdm(test_dataloader):

        word_ids = batch["word_ids"].detach().cpu().tolist()

        cuis = batch["cuis"].detach().cpu().tolist()

        del batch["word_ids"]

        del batch["cuis"]

        batch.to(device)

        with torch.no_grad():

            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1).detach().cpu().tolist()

        labels = batch["labels"].detach().cpu().tolist()

        tokens = batch["input_ids"].detach().cpu().tolist()

        true_predictions, true_labels = postprocess(predictions, labels, label_names)
        metric.add_batch(predictions=true_predictions, references=true_labels)
        
        for i in range(len(tokens)):

            sentence = tokens[i]

            sentence_word_ids = word_ids[i]

            sentence_predictions = predictions[i]

            sentence_labels = labels[i]

            sentence_cuis = cuis[i]

            num_words = max(sentence_word_ids) + 1

            words = [""] * num_words

            word_level_sentence_predictions = []

            word_level_sentence_labels = []

            word_level_cuis = []

            for token, word_id, token_prediction, token_label, sentence_cui in zip(sentence, sentence_word_ids, sentence_predictions, sentence_labels, sentence_cuis):

                if word_id != -100:

                    token_string = tokenizer.convert_ids_to_tokens(token)

                    if len(words[word_id]) == 0:
                        word_level_sentence_predictions.append(token_prediction)
                        word_level_sentence_labels.append(token_label)
                        word_level_cuis.append(sentence_cui)

                    words[word_id] += token_string
            
            words = [x.replace("##", "") for x in words]

            total_tokens.extend(words)

            total_labels.extend(word_level_sentence_labels)

            total_predictions.extend(word_level_sentence_predictions)

            total_cuis.extend(word_level_cuis)

            total_tokens.extend(" ")

            total_predictions.extend(" ")

            total_labels.extend(" ")

            total_cuis.extend(" ")

    results = metric.compute()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_predictions = [id2label[x] if x != " " else " " for x in total_predictions]

    total_labels = [id2label[x] if x != " " else " " for x in total_labels]

    total_cuis = [id2cui[x] if x != " " else " " for x in total_cuis]

    with open(conll_output, "w") as f:
        writer = csv.writer(f, delimiter=" ", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerows(zip(total_tokens, total_labels, total_predictions, total_cuis))

    print(
            f"test:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('-f','--files', nargs='+')
    parser.add_argument('-m','--model_checkpoint')
    parser.add_argument('-o','--model_output')
    parser.add_argument('-c','--conll_output')


    args = parser.parse_args()
    
    main(args)
