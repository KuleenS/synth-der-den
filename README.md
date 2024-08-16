# Using Text Generation Models for Normalization 

## Preperation

### Semeval 2015 Task 14
1. Follow steps to access the data here: 
```https://alt.qcri.org/semeval2015/task14/index.php?id=data-and-tools```
2. If the links to access does not work and you have access to the Aurora Research Computer, there is a tar.gz of it in this folder:
```/home/ksasse/semevaldata```
3. To prepare the data for T5 run the preprocess pipeline using the config with your semeval data (WIP)

### UMLS
You will need UMLS for the T5 data preperation
1. Follow this website to download UMLS locally: ```https://www.ncbi.nlm.nih.gov/books/NBK9683/``` 
2. The downloads are here: https://www.nlm.nih.gov/research/umls/licensedcontent/downloads.html
3. If you can access the Aurora Research Computer, set the credentials in your .env, source them and export them
```
UMLS_USER=<>
UMLS_PWD=<>
UMLS_DATABASE_NAME=<>
UMLS_IP=<>
```

## Env Preperation
Create conda env called `generation` from file with this command
```
conda env create --file environ.yml
```

### BERTNER Preperation
I know this is alot of envs for your machine \
If you have access to the UAB RC Gitlab and are in the informatics institute please clone https://gitlab.rc.uab.edu/oleaj/uabdeid with this command:
```
git clone https://gitlab.rc.uab.edu/oleaj/uabdeid.git
```
Follow the steps to install the uabdeid repo to that env
```
pip install -e /path/to/uabdeid
```
Now you can run all the test bench for bert training (WIP)

## Running Train 
After setting up environment, to run training
```bash
python -m src --config config.toml
```

The config.toml consists of one section currently `train`. It has parameters

1. `training_method`: `str`: unsupervised or supervised
2. `model`: `str` : Path to Model folder to load into huggingface or name of model
3. `output_folder`: `str` : Path of place to store checkpoints
4. `special_token_path`: `str` : Optional : Path to txt file with tokens every line that need replacing in text before preprocessing
5. `extra_token_path`: `str` : Optional : Path to txt file with tokens will be added to tokenizer
6. `train_data` : `str` : Path to csv file for training
    - File format for unsupervised is one column of text. Each training item is a line
    - File format for supervised is two column csv, one prompt one output
7. `eval_data`: `str` : Path to csv file for evaluation
    - File format for unsupervised is one column of text. Each training item is a line
    - File format for supervised is two column csv, one prompt one output
8. `batch_size`: `int`: batch size for model
9. `learning_rate`: `float`: Learning Rate of Optimizer
10. `epochs`: `int`: Epochs to train file
11. `warmup_steps` : `int` : Learning Rate Warmup Steps 
12. `external_logger`: `bool`: whether to use wandb (**MUST HAVE `WANDB_API_KEY` IN ENV VARIABLES**)
13. `logging_project`: `str`: name of wandb project

## Running Preprocess (WIP)
Still in development
## Running Evaluate (WIP)
Still in development








