import argparse

import logging

import tomli

from datasets import Dataset

from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments

import torch

from peft import LoraConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM 

from preprocess import SupervisedDataPreprocess

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['source'])):
        text = f"### Instruction: {example['source'][i]}\n ### Example Doctor's Note: {example['target'][i]}"
        output_texts.append(text)
    return output_texts

def main(args):

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    train_config = config['train-llama']

    model_path = train_config['model']
    model_output = train_config['output_folder']

    special_token_path = train_config['special_token_path']
    extra_token_path = train_config['extra_token_path']

    train_data = train_config['train_data']
    eval_data = train_config['eval_data']

    lr = train_config['learning_rate']

    num_epochs = train_config['epochs']

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding=True, truncation=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = "right"  # Fix for fp16

    train_dataset_processor = SupervisedDataPreprocess(train_data, tokenizer, special_token_path, extra_token_path)
    eval_dataset_processor = SupervisedDataPreprocess(eval_data, tokenizer, special_token_path, extra_token_path)

    train_source, train_target = train_dataset_processor.process_files()
    eval_source, eval_target = eval_dataset_processor.process_files()

    train_dataset = Dataset.from_dict({"source": train_source, "target": train_target})

    eval_dataset = Dataset.from_dict({"source": eval_source, "target": eval_target})

    response_template = " ### Example Doctor's Note:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    model_kwargs = dict(
        attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


    training_args = TrainingArguments(
        bf16=True, # specify bf16=True instead when training on GPUs that support bf16
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        log_level="info",
        logging_steps=5,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        max_steps=-1,
        num_train_epochs=num_epochs,
        output_dir=model_output,
        overwrite_output_dir=True,
        save_steps=1000,
        per_device_train_batch_size=4,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=tokenizer.model_max_length,
    )

    trainer.train()

    trainer.save_model(model_output)
    tokenizer.save_pretrained(model_output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)
