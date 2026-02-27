"""
config.py
─────────
Single source of truth for:
  - Environment variables (loaded from .env)
  - AWS Bedrock client
  - Langfuse v3 client
  - The Expenses subtree from COA_parsed.json (loaded once, shared read-only)
"""

import os
import boto3
from dotenv import load_dotenv
from langfuse import get_client, Langfuse
import asyncio
import redis
import redis.asyncio as redis_async



# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

mistral_semaphore = asyncio.Semaphore(5)

AWS_REGION=os.getenv("AWS_DEFAULT_REGION","ap-south-1")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRETE_KEY", "")
MODEL_ID=os.getenv("BEDROCK_MODEL_ID","google.gemma-3-12b-it")


# ── Logging / CloudWatch ──────────────────────────────────────────────────────
ENVIRONMENT            = os.getenv("ENVIRONMENT", "dev")
SERVICE_NAME           = os.getenv("SERVICE_NAME", "dr-agent")
LOG_CLOUDWATCH_ENABLED = os.getenv("LOG_CLOUDWATCH_ENABLED", "false").lower() == "true"
LOG_FILE_ENABLED       = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
CW_LOG_GROUP           = os.getenv("CW_LOG_GROUP", "/dr-agent/app")



DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# ── Constants from env ────────────────────────────────────────────────────────
AWS_REGION    = os.getenv("AWS_REGION", "ap-south-1")
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




# Session Management
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
REDIS_SSL_VERIFY = os.getenv("REDIS_SSL_VERIFY", "true").lower() == "true"

# Initialize Redis client for task queue
redis_client = None
redis_sync_client = None


if REDIS_ENABLED:
    try:
        redis_kwargs = {
            "decode_responses": True
        }

        # Only needed if SSL and you want to disable verification
        if REDIS_URL.startswith("rediss://") and not REDIS_SSL_VERIFY:
            redis_kwargs["ssl_cert_reqs"] = None

        redis_sync_client = redis.from_url(
            REDIS_URL,
            **redis_kwargs
        )

        redis_client = redis_async.from_url(
            REDIS_URL,
            **redis_kwargs
        )

        print(redis_sync_client.ping())

    except Exception as e:
        import logging
        logging.warning(f"Failed to connect to Redis: {e}")
        


