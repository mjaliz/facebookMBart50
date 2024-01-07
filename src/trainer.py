import os
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "..", 'data')

df = pd.read_csv(os.path.join(data_path, 'data.csv'))
df.rename(columns={'0': 'en', '1': 'fa'}, inplace=True)
df.drop(columns='Unnamed: 0', inplace=True)
df.dropna(inplace=True)
df["translation"] = df.apply(lambda row: {"en": row["en"], "fa": row["fa"]}, axis=1)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.01)

checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
#checkpoint = "/app/saved_model/facebook_finetuned/checkpoint-229860"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    src_lang="en_XX",
    tgt_lang="fa_IR")

source_lang = "en"
target_lang = "fa"


def preprocess_fn(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, padding=True, return_tensors="pt")
    return model_inputs


tokenized_dataset = dataset.map(preprocess_fn, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./../saved_model/facebook_finetuned",
    evaluation_strategy="steps",
    eval_steps=10000,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=50,
    predict_with_generate=True,
    fp16=True,
    save_strategy="epoch",
    save_steps=1,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
