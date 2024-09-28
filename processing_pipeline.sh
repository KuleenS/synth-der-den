INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

mkdir -p $OUTPUT_FOLDER/conll
mkdir -p $OUTPUT_FOLDER/labelled_conll

python pipeline/txt_to_conll.py -i $INPUT_FOLDER -o $OUTPUT_FOLDER/conll

python -m src.ner.predict.py -c $OUTPUT_FOLDER/labelled_conll -m biobert --files $OUTPUT_FOLDER/conll/*

python -m src.normalization_models.krissbert.predict --files $OUTPUT_FOLDER/labelled_conll/* --encoded_files prototypes/embeddings --entity_list_ids prototypes/cuis