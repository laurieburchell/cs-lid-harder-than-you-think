import logging
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

logging.basicConfig(level=logging.INFO)

# TODO: remove hardcoding
args = {
    "infile": "/home/laurie/projects/cs-lid-harder-than-you-think/data/train-data/lid200_arabicdialects_100124_shuf.fasttext",
    "outdir": "/home/laurie/projects/cs-lid-harder-than-you-think/models/marbert",
    "batch_size": 32
}

label2id = {
    "acw_Arab" : 0, 
    "aeb_Arab" : 1, 
    "afb_Arab" : 2, 
    "apc_Arab" : 3, 
    "arb_Arab" : 4, 
    "arq_Arab" : 5, 
    "ary_Arab" : 6, 
    "arz_Arab": 7}
id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)
MAX_LENGTH = 256  # from paper

tokeniser = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=NUM_LABELS,
                                                           id2label=id2label, label2id=label2id)
accuracy = evaluate.load("accuracy")

def load_data(infile, mini_dataset=False):
    """load data for finetuning in fasttext format"""
    logging.info(f"loading finetuning data from {infile}")
    with open(infile) as f:
        raw = [x.strip().replace('__label__', '').split(' ', 1) for x in tqdm(f.readlines(),
                                                                              desc="loading data from file")]
    data_dict = {'label': [label2id[x[0]] for x in raw],
                'text': [x[1] for x in raw]}
    df = pd.DataFrame.from_dict(data_dict)
    if mini_dataset:
        logging.info("loading mini dataset for quick development")
        df = df[:100000]
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=3000, seed=26)
    return ds

def tokenise_data(data, tokeniser):
    logging.info("tokenising text")
    def tokenise_fn(examples):
        return tokeniser(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    return data.map(tokenise_fn, batched=True, batch_size=10000)

def compute_accuracy(eval_pred):
    predictions, golds = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=golds)

def main():
    finetune_data = load_data(args['infile'], mini_dataset=True)
    tokenised_data = tokenise_data(finetune_data, tokeniser)
    training_args = TrainingArguments(output_dir=args['outdir'],
                                      evaluation_strategy="epoch",
                                      per_device_train_batch_size=args['batch_size'])
    
    print(training_args.device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_data['train'],
        eval_dataset=tokenised_data['test'],
        compute_metrics=compute_accuracy
    )

    trainer.train()

if __name__ == "__main__":
    main()