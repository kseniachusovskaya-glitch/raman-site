"""Собирает базу эталонов из data/raw/.

1. Положите исходные файлы спектров в data/raw/ (как скачаны или сняты).
2. Опишите каждый файл строкой в data/raw/metadata.csv.
   Эталоны RRUFF описаны в data/raw/metadata_rruff.csv
   (создаётся scripts/import_rruff.py).
3. Запустите: uv run python scripts/build_library.py

Открытые эталоны → data/library/ (лежит в git).
Свои эталоны из data/raw/metadata_own.csv (создаёт scripts/import_own.py)
→ data/library_own/ — эта папка в git НЕ попадает, а load_library
сам добавляет её к основной базе.
"""

from pathlib import Path

from ramancore.library import build_library

root = Path(__file__).parent.parent  # папка проекта raman-site
raw = root / "data" / "raw"
own_csv = raw / "metadata_own.csv"

# Все таблицы metadata*.csv, кроме своих: шаблон + RRUFF
tables = [t for t in sorted(raw.glob("metadata*.csv")) if t != own_csv]
n = build_library(metadata_csv=tables, raw_dir=raw, out_dir=root / "data" / "library")
print(f"Открытая база: {n} эталонов → data/library/")

if own_csv.exists():
    n = build_library(metadata_csv=own_csv, raw_dir=raw, out_dir=root / "data" / "library_own")
    print(f"Свои эталоны: {n} → data/library_own/ (не в git)")
