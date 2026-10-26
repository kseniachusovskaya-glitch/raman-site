"""Собирает базу эталонов из data/raw/.

Обычный запуск — собрать СВОИ эталоны:
    uv run python scripts/build_library.py
Берёт data/raw/metadata_own.csv (создаёт scripts/import_own.py)
→ data/library_own/. Эта папка в git НЕ попадает, а load_library сам
добавляет её к открытой базе data/library/.

Пересобрать открытую базу (RRUFF + data/raw/metadata.csv) — только если
распакованы архивы RRUFF (scripts/import_rruff.py):
    uv run python scripts/build_library.py --open
Готовая открытая база уже лежит в git, обычно пересобирать её не нужно.
"""

import sys
from pathlib import Path

from ramancore.library import build_library

root = Path(__file__).parent.parent  # папка проекта raman-site
raw = root / "data" / "raw"
own_csv = raw / "metadata_own.csv"

if "--open" in sys.argv:  # sys.argv — слова, написанные после имени скрипта
    # Все таблицы metadata*.csv, кроме своих: шаблон + RRUFF
    tables = [t for t in sorted(raw.glob("metadata*.csv")) if t != own_csv]
    n = build_library(metadata_csv=tables, raw_dir=raw, out_dir=root / "data" / "library")
    print(f"Открытая база: {n} эталонов → data/library/")

if own_csv.exists():
    n = build_library(metadata_csv=own_csv, raw_dir=raw, out_dir=root / "data" / "library_own")
    print(f"Свои эталоны: {n} → data/library_own/ (не в git)")
else:
    print("Нет data/raw/metadata_own.csv — сначала запустите scripts/import_own.py")
