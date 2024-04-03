import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd

df = pd.read_csv("data.csv")

text_col = []

for _, row, in df.iterrows():

    prompt = "Below is an instruction that describes a task, paired with an input that provided "
    instruction = str(row["instruction"])
    input_query = str(row["input"])
    response = str(row["output"])

    if len(input_query.strip()) == 0:
        text = prompt + '### Instruction: \n' + instruction + '\n### Response: \n' + response
    else:
        text = prompt + '### Instruction: \n' + instruction + "### Input: \n" + input_query + '\n### Response: \n' + response

    text_col.append(text)

df.lc[:, "text"] = text_col
print(df.head())

df.to_csv("train.csv", index=False)
