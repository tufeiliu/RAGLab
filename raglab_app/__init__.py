"""Streamlit app support: document collections, background jobs, logging and result views."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "models.yaml"
DATA = ROOT / "raglab_data"
COLLECTIONS = DATA / "collections"   # uploaded files and test sets, one folder per collection
INDEXES = DATA / "indexes"           # immutable vector indexes, keyed by files + settings
CACHE = DATA / "cache"               # embeddings.sqlite and responses.sqlite
JOBS = DATA / "jobs"                 # one folder per background job: status, log, report
LOGS = DATA / "logs"                 # app.log for interactive queries
