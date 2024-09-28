INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

mkdir -p $OUTPUT_FOLDER/conll
mkdir -p $OUTPUT_FOLDER/labelled_conll

python synth-der-den/pipeline/txt_to_conll.py -i $INPUT_FOLDER -o $OUTPUT_FOLDER/conll

python synth-der-den/src/ner/predict.py -c $OUTPUT_FOLDER/labelled_conll -m ??? --files $OUTPUT_FOLDER/conll/*