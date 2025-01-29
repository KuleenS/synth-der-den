# Using Text Generation Models for Normalization 

## Preperation

### Semeval 2015 Task 14
1. Follow steps to access the data here: 
```https://alt.qcri.org/semeval2015/task14/index.php?id=data-and-tools```

### UMLS
You will need UMLS
1. Follow this website to download UMLS locally: ```https://www.ncbi.nlm.nih.gov/books/NBK9683/``` 
2. The downloads are here: https://www.nlm.nih.gov/research/umls/licensedcontent/downloads.html
3. If you can access already have access, set the credentials in your .env, source them and export them
```
UMLS_USER=<>
UMLS_PWD=<>
UMLS_DATABASE_NAME=<>
UMLS_IP=<>
```

### Env Preperation

#### conda
Create conda env called `generation` from file with this command
```
conda env create --file environ.yml
```

#### pip (fixed)
To create an environment with the packages and their versions I used. First, create a conda env 
```
conda create -n generation python=3.10
pip install -r requirements-fixed.txt
```

#### pip (free)
To create an environment with the packages I used but not their versions. First, create a conda env 
```
conda create -n generation python=3.10
pip install -r requirements.txt
```

#### QuickUMLS
To set up quickumls for the environment follow [this](https://github.com/Georgetown-IR-Lab/QuickUMLS?tab=readme-ov-file#installation)

## Scripts 

### Data Preparation
To prepare data for LLM training run 
```
python -m src.preprocess.preprocess --config config.toml
```

In your `config.toml` file fill out this
```toml
[preprocess]
semeval_data = ""
semeval_output_data = ""
```
- `semeval_data`: input directory of semeval data
- `semeval_output_data`: output directory of preprocessed semeval data


### Training
To train LLM 

```
python -m src.training.train_llama --config config.toml
```

In your `config.toml` file fill out this
```toml
[train-llama]
model=""
output_folder=""
special_token_path = ""
extra_token_path = ""
train_data=""
eval_data=""
learning_rate =
epochs = 
```

- `model` : model name you want to train 
- `output_folder`: model save path 
- `special_token_path`: path to file with each line being a special token to add to the tokenizer
- `extra_token_path` : path to file with each line being a special token to remove from the texts
- `train_data`: path to train data folder csvs with two columns 
- `eval_data`: path to evaluation data folder csvs with two columns 
- `learning_rate` : floating point learning rate for model
- `epochs` : how many epochs to finetune

### Generation
To generate synthetic data run 

```
python -m src.generate.generate --config config.toml
```
In your `config.toml` file fill out this
```toml
[generate]
output_folder = ""
model_dir = ""
input_file = ""
[generate.generation_params]
model_name = ""
gpus = 
batch_size = 
examples_generated = 
max_length = 
temperature =
do_sample= 
```
- `output_folder`: output folder for generation
- `model_dir`: trained model folder
- `model_name`: trained model type
- `input_file`: Input file to generate
- `gpus`: number of gpus needed
- `batch_size`: batch size for model
- `examples_generated`: number of examples generated per input
- `max_length`: Max number of new tokens generated
- `temperature`: Temperature of sampling
- `do_sample`: Sample or Deterministic Generation


To clean up synthetic data run

```
python -m src.generate.postprocess_data <input> <output>
```
- `input`: input csv
- `output`: output csv

### Downstream Task

#### NER

First before running NER you must first split your semeval

```python src/split_semeval.py <semeval_folder> <output_semeval_split>```

To evaluate NER run 
```
python -m src.evaluate.evaluate --config config.toml
```

In your `config.toml` file fill out this
```toml
[evaluate]
generated_note_path = ""
results_output = ""
model_output=""
processed_dataset_output = ""
semeval_path = ""
mode = 0
```
- `generated_note_path`: path to generated nodes
- `results_output`: outputs for the results
- `model_output`: save dir for trained models
- `processed_dataset_output`: output path to save processed datasets
- `semeval_path`: Semeval Data Path Folder that is split from above
- `mode`: how to add the synthetic data
    - options: 
            -  No Synthetic = 0
            -  All Synthetic = 1
            -  Ideal Synthetic = 2
            -  Real World Synthetic = 3
            -  Ablation Synthetic = 4

To evaluate normalization run 
#### NEN

##### KrissBERT
To run the evaluation, you need to run two parts
```
python -m src.normalization_models.krissbert.generate_prototypes <semeval_data_path> <quick_umls_data>
```
- `dataset` which dataset (options: semeval, bc5dr, ncbi)
- `output`: output directory for embeddings
- `semeval_input`: path to semeval data that is split from above
- `generated_input`: path to synthetic data csv
- `mode`: which mode to add your data into the model 
    - options: 
        -  No Synthetic = 0
        -  All Synthetic = 1
        -  Ideal Synthetic = 2
        -  Real World Synthetic = 3
        -  Ablation Synthetic = 4

```
python -m src.normalization_models.krissbert.run_entity_linking --config nen_config.toml
```
In your `nen_config.toml` file fill out this for Semeval
```toml
# path to pretrained model and tokenizer
model = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
test_data= "<path to semeval data folder>"
dataset = "semeval"
# paths to encoded data
encoded_files= [
  "<path to embeddings generated by previous step>"
]

encoded_umls_files= []
entity_list_names= "<path to cuis generated by previous step>"
seed= 12345
batch_size= 256
max_length= 64
num_retrievals= 100
top_ks= [1, 5, 50, 100]
```

For other datasets fill out
```toml
model = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
dataset = "<dataset>"
# paths to encoded data
encoded_files= [
  "<path to embeddings generated by previous step>"
]

encoded_umls_files= []
entity_list_names= "<path to cuis generated by previous step>"
seed= 12345
batch_size= 256
max_length= 64
num_retrievals= 100
top_ks= [1, 5, 50, 100]
```

##### SapBERT
To run the evaluation
```
python -m src.normalization_models.sapbert.evaluate_sapbert <semeval_data_path> <quick_umls_data>
```
- `semeval_data_path`: path to semeval data
- `synthetic_data`: path to synthetic data csv


##### QuickUMLS
To evaluate QuickUMLS, you must first prepare the data for it like described in their repo. 

To run the evaluation
```
python -m src.normalization_models.quickumls.evaluate_quickumls <semeval_data_path> <quick_umls_data>
```
- `semeval_data_path`: path to semeval data
- `quick_umls_data`: path to processed quick umls data

##### SciSpacy
To run the evaluation
```
python -m src.normalization_models.scispacy.evaluate_scispacy.py <semeval_data_path>
```
- `semeval_data_path`: path to semeval data


## Pipeline/Inference

There are multple ways to do turn this into a pipeline method in your system: pure bash script and docker container

### Bash Script
To run the bash procsesing pipeline one must do 

1. Train BioBERT 
2. Train KrissBERT (python -m src.normalization_models.krissbert.generate_prototypes)
3. Run bash script `./processing_pipeline.sh <input_folder> <output_folder> <bert_folder> <krissbert_folder>`

### Docker Container

To run the Docker container one must do 

1. Train BioBERT 
2. Train KrissBERT (python -m src.normalization_models.krissbert.generate_prototypes)
3. Build the Docker `docker build -t synthderdenpipeline .`
4. Run the Docker  `docker run --gpus all -v $(pwd):/app/ synthderdenpipeline <input_folder> <output_folder> <bert_folder> <krissbert_folder>`

