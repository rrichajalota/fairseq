import os
from accelerate import infer_auto_device_map, init_empty_weights
os.environ['TRANSFORMERS_CACHE'] = '/netscratch/jalota/hf-cache/'
from datasets import load_dataset
datasets = load_dataset("text", data_files={"train": "/netscratch/jalota/datasets/haifa-hansard/train/original.txt", "validation":"/netscratch/jalota/datasets/haifa-hansard/dev/original.txt"})

# print(datasets["train"][0])
# print(datasets["train"][200])
# print(datasets["train"][10])
# print(datasets["train"][1000])
# print(datasets["train"][50])
# print(datasets["train"][100])

model_checkpoint = "gpt2"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# print(tokenized_datasets["train"][1])
block_size = 512

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=8000,
    num_proc=12,
)

# print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))

from transformers import AutoConfig, AutoModelForCausalLM
import torch

config = AutoConfig.from_pretrained(model_checkpoint)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", offload_folder="offload", offload_state_dict = True, torch_dtype=torch.float16)

device_map = infer_auto_device_map(model)

from transformers import Trainer, TrainingArguments
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    f"/netscratch/jalota/checkpoints/{model_name}-finetuned-CanadianHansardOriginals",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

print("started training")

trainer.train()
import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.save_model("/netscratch/jalota/checkpoints/huggingface-gpt2-3/")


