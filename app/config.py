import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None) -> dict:
    path = Path(path or os.environ.get("TRAFFIC_CONFIG", ROOT / "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
