import csv
import json
from pathlib import Path
from typing import Dict, List

from retrieval.Cleantext import clean_text


def preprocess_inventory(csv_path: str) -> List[Dict]:
    records: List[Dict] = []
    with Path(csv_path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["clean_description"] = clean_text(row.get("description", ""))
            row["document_type"] = "inventory"
            records.append(row)
    return records


def preprocess_promotions(json_path: str) -> List[Dict]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    for item in data:
        item["clean_description"] = clean_text(item.get("description", ""))
        item["document_type"] = "promotions"
    return data


def preprocess_faq(json_path: str) -> List[Dict]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    for item in data:
        item["clean_question"] = clean_text(item.get("question", ""))
        item["clean_answer"] = clean_text(item.get("answer", ""))
        item["document_type"] = "general"
    return data


def preprocess_tickets(json_path: str) -> List[Dict]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    for item in data:
        item["clean_subject"] = clean_text(item.get("subject", ""))
        item["clean_description"] = clean_text(item.get("description", ""))
        item["document_type"] = "ticket"
    return data
