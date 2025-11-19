import json
import os

BASE_DIR = "/app/artifacts"
EXPECTED_COLS_PATH = os.path.join(BASE_DIR, "expected_columns.json")

with open(EXPECTED_COLS_PATH) as f:
    EXPECTED_COLS = list(json.load(f).keys())
