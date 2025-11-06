import os
import json
from datasets import Dataset
import polars as pl
from datasets import load_dataset

def load_finqa_from_disk(json_path: str):
    """
    Load a FinQA JSON file (which is an array of JSON objects),
    return a Hugging Face `datasets.Dataset`.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transformed = []
    for ex in data:
        qa = ex.get("qa", {})
        transformed.append({
            "id": ex.get("id"),
            "pre_text": ex.get("pre_text"),
            "post_text": ex.get("post_text"),
            "table": ex.get("table"),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "program": qa.get("program"),
            "steps": qa.get("steps")
        })
    return Dataset.from_list(transformed)


## source for wikipedia data
# https://huggingface.co/datasets/lucadiliello/english_wikipedia
# maybe not essential to download it locally if ai labs work

def download_wikipedia_data():
    ds = load_dataset("Salesforce/wikitext",  "wikitext-2-raw-v1")
    df = pl.DataFrame(ds['train'])
    train = ds["train"]

    df = pl.DataFrame(train.to_list())
    df.write_csv("wikipedia_data.csv")
    return df
