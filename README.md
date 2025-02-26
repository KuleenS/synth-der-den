# Using Text Generation Models for Normalization 

## Preperation

### Semeval 2015 Task 14
1. Follow steps to access the data here: 
```https://alt.qcri.org/semeval2015/task14/index.php?id=data-and-tools```

### UMLS
You will need UMLS
1. Follow this website to download UMLS locally: ```https://www.ncbi.nlm.nih.gov/books/NBK9683/``` 
2. The downloads are here: https://www.nlm.nih.gov/research/umls/licensedcontent/downloads.html
   
If you can access UMLS and have it set up, set the credentials in your `.env`, source them and export them
```
UMLS_USER=<>
UMLS_PWD=<>
UMLS_DATABASE_NAME=<>
UMLS_IP=<>
```

You can export your environment variables using 

```bash
export $(grep -v '^#' .env | xargs)
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
semeval_data = "semeval/"
semeval_output_data = "processed_semeval/"
```
- `semeval_data`: Path - input directory of semeval data
- `semeval_output_data`: Path - output directory of preprocessed semeval data


### Training
To train LLM 

```
python -m src.training.train_llama --config config.toml
```

In your `config.toml` file fill out this
```toml
[train-llama]
model="meta-llama/Llama-2-13b"
output_folder="output/"
special_token_path = "example/special_tokens.txt"
extra_token_path = "example/extra_tokens.txt"
train_data="processed_semeval/traindata"
eval_data="processed_semeval/devdata"
learning_rate = 2e-5
epochs = 5
```

- `model` : str -  model name you want to train 
- `output_folder`: Path - model save path 
- `special_token_path`: Path - path to file with each line being a special token to add to the tokenizer
- `extra_token_path` : Path - path to file with each line being a special token to remove from the texts
- `train_data`: Path - path to train data folder csvs with two columns 
- `eval_data`: Path - path to evaluation data folder csvs with two columns 
- `learning_rate` : float - floating point learning rate for model
- `epochs` : int - how many epochs to finetune

### Generation
To generate synthetic data run 

```
python -m src.generate.generate --config config.toml
```
In your `config.toml` file fill out this
```toml
[generate]
output_folder = "generation_output/"
model_dir = "output/"
input_file = ""
[generate.generation_params]
gpus = 1
batch_size = 16
examples_generated = 5
max_length = 256
temperature = 0.7
do_sample= true
```
- `output_folder`: Path - output folder for generation
- `model_dir`: Path - trained model folder
- `input_file`: Path - Input file to generate (can be a file that does not exist)
- `gpus`: int - number of gpus needed
- `batch_size`: int - batch size for model
- `examples_generated`: int - number of examples generated per input
- `max_length`: int - Max number of new tokens generated
- `temperature`: float - Temperature of sampling
- `do_sample`: bool - Sample or Deterministic Generation

### Downstream Task

#### NER

First before running NER/NEN you must first split your semeval

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
- `generated_note_path`: Path - path to postprocessed generated csv (from src.generate.generate)
- `results_output`: Path - outputs for the results
- `model_output`: Path - save dir for trained models
- `processed_dataset_output`: Path - output path to save processed datasets
- `semeval_path`: Path - Semeval Data Path Folder that is split from above
- `mode`: int - how to add the synthetic data
    - options: 
        -  No Synthetic = 0
        -  Naive Synthetic = 1
        -  Ideal Synthetic = 2
        -  Supplemental Synthetic = 3
        -  Ablation Synthetic = 4

To evaluate normalization run 
#### NEN

##### KrissBERT
To run the evaluation, you need to run two parts
```
python -m src.normalization_models.krissbert.generate_prototypes <dataset> <output> <semeval_input> <generated_input> <mode>
```
- `dataset` : str - which dataset (options: semeval, bc5dr, ncbi)
- `output`: Path - output directory for embeddings
- `semeval_input`: Path - path to semeval data that is split from above
- `generated_input`: Path - path to postprocessed synthetic data csv
- `mode`: int - which mode to add your data into the model 
    - options: 
        -  No Synthetic = 0
        -  Naive Synthetic = 1
        -  Ideal Synthetic = 2
        -  Supplemental Synthetic = 3
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
python -m src.normalization_models.sapbert.evaluate_sapbert <semeval_data_path> <synthetic_data>
```
- `semeval_data_path`: Path - path to semeval data
- `synthetic_data`: Path - path to postprocessed synthetic data csv


##### QuickUMLS
To evaluate QuickUMLS, you must first prepare the data for it like described in their repo. 

To run the evaluation
```
python -m src.normalization_models.quickumls.evaluate_quickumls <semeval_data_path> <quick_umls_data>
```
- `semeval_data_path`: Path -  path to semeval data
- `quick_umls_data`: Path -  path to processed quick umls data

##### SciSpacy
To run the evaluation
```
python -m src.normalization_models.scispacy.evaluate_scispacy.py <semeval_data_path>
```
- `semeval_data_path`: Path -  path to semeval data


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

