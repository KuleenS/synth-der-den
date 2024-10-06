INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

mkdir -p $OUTPUT_FOLDER/labelled_conll

python -m src.ner.pipeline_predict -c $OUTPUT_FOLDER/labelled_conll -m biobert --files $INPUT_FOLDER/*

python -m src.normalization_models.krissbert.predict --files $OUTPUT_FOLDER/labelled_conll/* --encoded_files prototypes/embeddings --entity_list_ids prototypes/cuis