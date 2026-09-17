import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq's hosted model lineup changes over time (models get decommissioned) — if
# this 404s with "model does not exist", check https://console.groq.com/docs/models
# or `client.models.list()` for the current catalog. openai/gpt-oss-120b is the
# strongest general-purpose text model on the free tier as of writing;
# openai/gpt-oss-20b is a smaller/faster fallback.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Used if GROQ_MODEL 404s (decommissioned) — the draft/fact-check calls retry
# once against this model before giving up.
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

# fact_check_node only needs to extract JSON, not write prose — a smaller/faster
# model is fine here and, as a bonus, spreads token usage across two separate
# Groq per-model rate-limit buckets instead of hammering one.
GROQ_FACT_CHECK_MODEL = os.getenv("GROQ_FACT_CHECK_MODEL", "openai/gpt-oss-20b")

MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

# When true, unhandled exceptions are returned to the client as JSON with the
# full traceback instead of a bare "Internal Server Error". Defaults on for
# local dev; set DEBUG=false (or unset) in production.
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")
