# %%
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
import numpy as np


# hyperparameters from https://github.com/clinc/oos-eval/blob/master/hyperparameters.csv
learning_rate = 4.00e-05
warmup_proportion = 0.1
train_batch_size = 32
eval_batch_size = 32
num_train_epochs = 5
gradient_accumulation_steps = 4

huggingface_modelname = "bert-base-uncased"

TRAIN_SAMPLE_SIZE = int(10e4)
TEST_SAMPLE_SIZE = int(10e3)
OUTPUT_DIR = "/var/scratch/fht800/hf_home"

dbpedia = load_dataset('fancyzhx/dbpedia_14')
dataset = dbpedia

dataset

tokenizer = AutoTokenizer.from_pretrained(huggingface_modelname)

def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset['train'].features['label'].names, dataset

labels = tokenized_datasets['train'].features['label'].names
num_labels = len(labels)
id2label = {k: v for k,v in enumerate(labels)}
label2id = {v: k for k,v in id2label.items()}

label2id

model = AutoModelForSequenceClassification.from_pretrained(
    huggingface_modelname,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=gradient_accumulation_steps,
    load_best_model_at_end=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("saved model")

# TODO: evaluate final model on test set
print(model.eval())
print(trainer.evaluate())