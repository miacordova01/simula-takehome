"""Central paths and shared constants."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw inputs. Override with SIMULA_DATA_DIR if the CSVs live elsewhere.
DATA_DIR = Path(os.environ.get("SIMULA_DATA_DIR", Path.home() / "Downloads"))
IMPRESSIONS_CSV = DATA_DIR / "impressions.csv"
CHARACTERS_CSV = DATA_DIR / "characters.csv"

# Derived / output locations.
CACHE = ROOT / "data"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

for _p in (CACHE, REPORTS, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# Column groups.
ID_COL = "id"
TARGET = "click"
TIME_COL = "hour"

CHAR_KEY = "character_id"

# Ad-side ("creative / advertiser") attributes: these are what a retrieval layer
# varies across candidates. Everything else describes the request context.
# C14/C17 are the high-cardinality creative-ish ids in this schema; C18-C21 and
# banner_pos behave like slot/creative attributes.
AD_FEATURES = ["banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]

# Context ("request") attributes: fixed for a given ad opportunity.
CONTEXT_FEATURES = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C1",
    "character_id",
    "conversation_turn",
    "session_msg_count",
]

# The sentinel value Avazu-style logs use for "no device cookie".
UNKNOWN_DEVICE_ID = "a99f214a"

RANDOM_SEED = 17
