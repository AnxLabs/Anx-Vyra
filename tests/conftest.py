"""Test-session setup for the private Kaggle config.

The real ``kaggle_config.json`` is gitignored. When it is absent (a clean clone, CI, the publisher's staging checkout)
the committed placeholder template is used so the unit tests still run; the builders refuse the template unless
``VYRA_KAGGLE_ALLOW_EXAMPLE=1``, which is set here only for that fallback. A real config, or an explicit
``VYRA_KAGGLE_CONFIG``, is never overridden.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if not os.environ.get("VYRA_KAGGLE_CONFIG") and not (ROOT / "kaggle_config.json").is_file():
    os.environ["VYRA_KAGGLE_CONFIG"] = str(ROOT / "kaggle_config.example.json")
    os.environ["VYRA_KAGGLE_ALLOW_EXAMPLE"] = "1"
