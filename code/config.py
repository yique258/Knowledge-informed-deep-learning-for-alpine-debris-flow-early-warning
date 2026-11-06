"""Path configuration for the released code.

All data files are resolved relative to the repository root, so the code runs
from a fresh clone without editing absolute paths.  Set ``DF_DATA_DIR`` to
point somewhere else (for example a local copy of the ERA5 extract).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DF_DATA_DIR", REPO_ROOT / "data"))
RESULT_DIR = Path(os.environ.get("DF_RESULT_DIR", REPO_ROOT / "results"))

DS_PLUS = DATA_DIR / "ds_plus.csv"          # hourly ERA5-Land meteorology, 2022-2025
INVENTORY = DATA_DIR / "inventory.csv"      # documented DF events (UTF-8)
RAIN = DATA_DIR / "rain.csv"                # rainfall-triggered events
HEAT = DATA_DIR / "heat.csv"                # melt-triggered events
DS_2026 = DATA_DIR / "ds_one_2026.csv"      # hourly ERA5-Land meteorology, 2026

RESULT_DIR.mkdir(parents=True, exist_ok=True)
