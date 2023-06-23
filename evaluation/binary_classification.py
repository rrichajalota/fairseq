import torch
from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer, BertForSequenceClassification
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import disable_caching, Dataset
disable_caching()
import logging
logging.disable(logging.INFO)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, use_fast=True) # True
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

labels = ClassLabel(names=['0', '1'])

def preprocess_function(examples):
    result = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    result['labels'] = [labels.str2int(str(label)) if label is not None else None for label in examples["label"]]

    return result

def tokenize_text(examples):
    result = tokenizer(str(examples["text"]),truncation=True,  max_length=512, padding='max_length', return_overflowing_tokens=True)

    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run binary classifer')
    parser.add_argument("--train", default="/netscratch/anonymous/datasets/motra-preprocessed/en_de/train/train_bt_bal.tsv") # based on *.tok.norm.true.txt - equal examples in both files!
    parser.add_argument("--dev", default="/netscratch/anonymous/datasets/motra-preprocessed/en_de/dev/dev_bt.tsv")  # based on translated.tok.norm.true.txt and original.tok.norm.true.txt -- equal examples in both files! 
    parser.add_argument("--test", default="/netscratch/anonymous/datasets/motra-preprocessed/en_de/test/test_bt_bal.tsv") # based on translated.tok.norm.true.txt and original.tok.norm.true.txt - equal examples in both files!
    parser.add_argument("--model", default=None)
    parser.add_argument("--out_dir", default="/netscratch/anonymous/results/binaryClassification_balanced_bt_og")
    args = parser.parse_args()
    # https://discuss.huggingface.co/t/using-trainer-at-inference-time/9378/7

    print(args.test)
    print(args.model)

    if args.model:
        test_df = pd.read_csv(args.test, delimiter="\t", names=['text', 'label'], quoting=csv.QUOTE_NONE)
        test_dataset = Dataset.from_pandas(test_df)
    else:
        dataset = load_dataset("csv", delimiter="\t", column_names=['text', 'label'], data_files={"train": args.train, "test": args.test, "dev": args.dev}) # streaming=True
    batch_size = 16
    metric_name = "accuracy" # "f1" 
    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    if args.model is None:
        # print(dataset["train"]['text'])
        encoded_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding='max_length', max_length=512), batched=True, batch_size=2000) # for streaming dataset which is an IterableDataset
        # encoded_dataset = dataset.map(preprocess_function, batched=True, num_proc=30)
        print("done with encoding")

        args = TrainingArguments(
        args.out_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        max_steps=1000,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        save_total_limit = 2,
        push_to_hub=False,)

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.evaluate()
        print(trainer.predict(encoded_dataset["test"]))
        savepath = f"{args.out_dir}/saved_model/"
        Path(savepath).mkdir(parents=True, exist_ok=True)
        trainer.save_model(savepath)
        # "/netscratch/anonymous/checkpoints/binaryClassification_balanced_bt_og/"

    else:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        # for motra -- uncomment the line below with lambda x 
        # encoded_dataset = test_dataset.map(lambda x: tokenizer(str(x["text"]), truncation=True, padding='max_length', max_length=512), batched=True, batch_size=2000)
        # encoded_dataset = dataset.map(preprocess_function, batched=True)
        encoded_dataset = test_dataset.map(tokenize_text, batched=True, batch_size=100)
        encoded_dataset = encoded_dataset.filter(lambda example: example['label'] is not None)
        #dataset["test"].map(preprocess_function, batched=True)
        # print(f"encoded_dataset['test']['label']: {encoded_dataset['test']['label']}")
        model = BertForSequenceClassification.from_pretrained(args.model)
        
        # arguments for Trainer
        test_args = TrainingArguments(
            output_dir = args.out_dir,
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size = batch_size,   
            dataloader_drop_last = False    
        )

        # init trainer
        trainer = Trainer(
                    model = model, 
                    args = test_args, 
                    compute_metrics = compute_metrics)

        test_results = trainer.predict(encoded_dataset) # ["test"]
        print(test_results)



# i think not .
# Mr President .
# Thank you .
# Thank you for your attention .
# That is wrong .

# !
# But it is not .
# co-rapporteur .
# i ask for your support .
# i do not believe so .
# i do not know .
# i just wanted to point that out .
# It does not .
# It is as simple as that .
# It is not .
# i welcome that .
# Let me give you an example .
# Let me now turn to the amendments .
# Mr President .
# Mr President , on a point of order .
# Mr President , on a point of order .
# No !
# No .
# No .
# Nothing .
# Our position is quite clear .
# So far , so good .
# Thank you .
# Thank you .
# Thank you , Commissioner .
# Thank you for your attention .
# Thank you for your cooperation .
# Thank you , Mr President .
# Thank you very much .
# Thank you very much .
# Thank you very much , Mr President .
# That is going too far .
# That is my first point .
# That is my first point .
# That is not acceptable .
# That is not acceptable .
# That is not correct .
# That is not the case .
# That is not the case .
# That is right and proper .
# That is the first issue .
# That is the objective .
# That is the reality .
# That is very important .
# That is what this is about .
# The Commission is the guardian of the Treaties .
# The Commission welcomes this .
# The list goes on .
# There is still much to be done , however .
# The same applies to the European Union .
# This is completely unacceptable .
# This is not acceptable .
# This is unacceptable .
# This is unacceptable .
# This is unacceptable .
# We all know that .
# We disagree .
# We know that .
# We know that .
# We should not forget that .
# Why ?
# Why ?
# Why ?
# Why ?
# Why ?
# Why ?
# Why ?
# Why is that the case ?
# Why is this so ?
# Why is this so important ?
# Why not ?
