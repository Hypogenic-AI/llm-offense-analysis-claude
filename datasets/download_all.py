"""
Download all datasets for LLM Offense Detection research.

Usage:
    source .venv/bin/activate
    python datasets/download_all.py
"""

import itertools
import json
import os

from datasets import Dataset, load_dataset

BASE = os.path.dirname(os.path.abspath(__file__))


def save_sample(data, path, n=10):
    sample = [dict(ex) for ex in list(data)[:n]]
    with open(path, "w") as f:
        json.dump(sample, f, indent=2, default=str)


def download_civil_comments(n=10_000):
    print(f"Downloading Civil Comments ({n} examples)...")
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    subset = list(itertools.islice(ds, n))
    out = Dataset.from_list(subset)
    out.save_to_disk(f"{BASE}/civil_comments/data")
    save_sample(subset, f"{BASE}/civil_comments/sample.json")
    print(f"  Done: {len(subset)} examples. Fields: {list(subset[0].keys())}")


def download_toxigen():
    print("Downloading ToxiGen (full)...")
    ds = load_dataset("skg/toxigen-data", name="train", split="train")
    ds.save_to_disk(f"{BASE}/toxigen/data")
    save_sample(ds, f"{BASE}/toxigen/sample.json")
    print(f"  Done: {len(ds)} examples. Fields: {list(ds.features.keys())}")


def download_real_toxicity_prompts(n=10_000):
    print(f"Downloading RealToxicityPrompts ({n} examples)...")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
    subset = list(itertools.islice(ds, n))
    out = Dataset.from_list(subset)
    out.save_to_disk(f"{BASE}/real_toxicity_prompts/data")
    save_sample(subset, f"{BASE}/real_toxicity_prompts/sample.json")
    print(f"  Done: {len(subset)} examples.")


def download_emotion():
    print("Downloading Emotion (full)...")
    ds = load_dataset("dair-ai/emotion")
    ds.save_to_disk(f"{BASE}/emotion/data")
    save_sample(ds["train"], f"{BASE}/emotion/sample.json")
    print(f"  Done: {ds}")


def download_go_emotions():
    print("Downloading GoEmotions (full)...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    ds.save_to_disk(f"{BASE}/go_emotions/data")
    save_sample(ds["train"], f"{BASE}/go_emotions/sample.json")
    print(f"  Done: {ds}")


def download_xstest():
    print("Downloading XSTest (full)...")
    ds = load_dataset("natolambert/xstest-v2-copy")
    ds.save_to_disk(f"{BASE}/xstest/data")
    save_sample(ds["prompts"], f"{BASE}/xstest/sample.json")
    print(f"  Done: {ds}")


def download_truthful_qa():
    print("Downloading TruthfulQA (full)...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    ds.save_to_disk(f"{BASE}/truthful_qa/data")
    save_sample(ds["validation"], f"{BASE}/truthful_qa/sample.json")
    print(f"  Done: {ds}")


if __name__ == "__main__":
    download_civil_comments()
    download_toxigen()
    download_real_toxicity_prompts()
    download_emotion()
    download_go_emotions()
    download_xstest()
    download_truthful_qa()
    print("\nAll datasets downloaded successfully.")
