"""
config.py
─────────
Single source of truth for:
  - Environment variables (loaded from .env)
  - AWS Bedrock client
  - Langfuse v3 client
  - The Expenses subtree from COA_parsed.json (loaded once, shared read-only)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import boto3
from dotenv import load_dotenv
from langfuse import get_client, Langfuse

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── Constants from env ────────────────────────────────────────────────────────
AWS_REGION    = os.getenv("AWS_REGION", "ap-south-1")
MODEL_ID      = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
COA_JSON_PATH = os.getenv("COA_JSON_PATH", "./COA_parsed.json")

# ── AWS Bedrock client (shared, thread-safe for reads) ────────────────────────
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# ── Langfuse v3 client ────────────────────────────────────────────────────────
# Reads LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST from env
# get_client() returns the globally initialized singleton after first call
Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)
langfuse = get_client()


# ── Load & extract Expenses subtree ───────────────────────────────────────────
def _load_expenses_tree(path: str) -> Dict[str, Any]:
    """
    Load COA_parsed.json and return the 'Expenses' sub-dict.
    Raises FileNotFoundError if path is wrong.
    Raises KeyError if 'Expenses' is not found in hierarchy.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"COA JSON not found at: {resolved}")

    with open(resolved) as f:
        raw = json.load(f)

    hierarchy: Dict[str, Any] = raw.get("hierarchy", {})

    # # Case-insensitive match for "Expenses" top-level key
    # pattern = re.compile(r"(?i)^\s*expenses\s*$")
    # for key, value in hierarchy.items():
    #     if pattern.match(key):
    #         if not isinstance(value, dict) or not value:
    #             raise ValueError(f"Expenses key '{key}' is empty or not a dict.")
    #         return value

    return hierarchy

    # raise KeyError("No 'Expenses' key found in COA hierarchy.")


# Loaded once at import time — shared read-only across all threads
EXPENSES_TREE: Dict[str, Any] = _load_expenses_tree(COA_JSON_PATH)
