"""Собирает базу эталонов из data/raw/ в data/library/.

1. Положите исходные файлы спектров в data/raw/ (как скачаны или сняты).
2. Опишите каждый файл строкой в data/raw/metadata.csv.
3. Запустите: uv run python scripts/build_library.py
"""

from pathlib import Path

from ramancore.library import build_library

root = Path(__file__).parent.parent  # папка проекта raman-site
n = build_library(
    metadata_csv=root / "data" / "raw" / "metadata.csv",
    raw_dir=root / "data" / "raw",
    out_dir=root / "data" / "library",
)
print(f"База собрана: {n} эталонов → data/library/")
