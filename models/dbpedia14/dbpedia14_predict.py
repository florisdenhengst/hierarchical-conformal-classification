from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import scipy
import pandas as pd

# TODO: move to multiple seeds loop
np.random.seed(42)

TRAIN_SAMPLE_SIZE = int(10e4)
TEST_SAMPLE_SIZE = int(10e3)
OUTPUT_DIR = "/var/scratch/fht800/hf_home"

dbpedia = load_dataset('fancyzhx/dbpedia_14', split='test')
dataset = dbpedia

# TODO FdH: split test set into calibration and evaluation sets

huggingface_modelname = "florisdh/bert-base-dbpedia14"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

labels = tokenized_dataset.features['label'].names
num_labels = len(labels)
id2label = {k: v for k,v in enumerate(labels)}
label2id = {v: k for k,v in id2label.items()}


model = AutoModelForSequenceClassification.from_pretrained(
    huggingface_modelname,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    )

trainer = Trainer(
    model=model,
    train_dataset=None,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

predictions = trainer.predict(tokenized_dataset)
logits = predictions.predictions
softmaxes = scipy.special.softmax(logits, axis=1)
np.testing.assert_allclose(softmaxes.sum(axis=1), 1.0, rtol=1e-6)

sm_dict = {f"label_{i}": softmaxes[:,i] for i in range(softmaxes.shape[1])}
frame = pd.DataFrame(
    {"content": dataset["content"],
     "label":  dataset["label"],
     **sm_dict
    }
)
frame.to_csv("data/dbpedia14_bert.csv", index_label='id')