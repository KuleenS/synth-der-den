#!/bin/bash

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
BERT_FOLDER="$3"
KRISSBERT_FOLDER="$4"


mkdir -p $OUTPUT_FOLDER/labelled_conll
mkdir -p $OUTPUT_FOLDER/ann_files

python -m src.ner.pipeline_predict -c $OUTPUT_FOLDER/labelled_conll -m $BERT_FOLDER --files $INPUT_FOLDER/*

python -m src.normalization_models.krissbert.predict --output_dir $OUTPUT_FOLDER/ann_files --files $OUTPUT_FOLDER/labelled_conll/* --encoded_files $KRISSBERT_FOLDER/embeddings --entity_list_ids $KRISSBERT_FOLDER/cuis