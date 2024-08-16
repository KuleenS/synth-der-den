import argparse

import logging

import tomli

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.training.preprocess import SupervisedDataPreprocess, UnsupervisedDataPreprocess

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

def main(args):

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    train_config = config['train']

    model_path = train_config['model']
    model_output = train_config['output_folder']

    special_token_path = train_config['special_token_path']
    extra_token_path = train_config['extra_token_path']

    train_data = train_config['train_data']
    eval_data = train_config['eval_data']

    lr = train_config['learning_rate']

    num_epochs = train_config['epochs']

    num_warmup_steps = train_config['warmup_steps']

    training_method = train_config['training_method']

    model_max_length = train_config['model_max_length']

    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.3,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length = model_max_length)

    if training_method == "supervised":

        train_dataset_processor = SupervisedDataPreprocess(train_data, tokenizer, special_token_path, extra_token_path)
        eval_dataset_processor = SupervisedDataPreprocess(eval_data, tokenizer, special_token_path, extra_token_path)

        train_source, train_target = train_dataset_processor.process_files()
        eval_source, eval_target = eval_dataset_processor.process_files()

        train_dataset = train_dataset_processor.tokenize((train_source,train_target))
        eval_dataset = eval_dataset_processor.tokenize((eval_source, eval_target))
    
    elif training_method == "unsupervised":
        train_dataset_processor = UnsupervisedDataPreprocess(train_data, tokenizer, special_token_path, extra_token_path)
        eval_dataset_processor = UnsupervisedDataPreprocess(eval_data, tokenizer, special_token_path, extra_token_path)

    
        train_target = train_dataset_processor.process_files()
        eval_target = eval_dataset_processor.process_files()

        train_dataset = train_dataset_processor.tokenize(train_target)
        eval_dataset = eval_dataset_processor.tokenize(eval_target)
        
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output,
        learning_rate=lr,
        auto_find_batch_size=True,
        num_train_epochs=num_epochs,
        warmup_steps=num_warmup_steps,
        logging_dir=f"{model_output}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="wandb",
        overwrite_output_dir = True,
        save_total_limit=3,
    )

    model.config.use_cache = False

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    trainer.model.save_pretrained(model_output)
    tokenizer.save_pretrained(model_output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)
