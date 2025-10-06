import os
import json
from datasets import Dataset
import polars as pl


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

