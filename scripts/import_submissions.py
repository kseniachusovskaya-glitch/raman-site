"""Добавляет принятые заявки коллег в открытую базу.

Заявки приходят с сайта в закрытый репозиторий (папка submissions/<id>/:
spectrum.txt, metadata.json, original_*). Как принять заявки:
  1. Скачайте (git clone / git pull) закрытый репозиторий заявок.
  2. Посмотрите спектры и описания. Принятые заявки перенесите в папку
     accepted/ того же репозитория, отклонённые просто удалите.
  3. Запустите из папки проекта raman-site:
         uv run python scripts/import_submissions.py "<путь>/raman-submissions/accepted"
  4. Закоммитьте data/raw/contrib/, data/raw/metadata_contrib.csv и
     data/library_contrib/ — они открытые (лицензия CC BY 4.0).

Повторный запуск безопасен: заявки, которые уже есть в базе, пропускаются.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from ramancore.library import build_library
from ramancore.submission import library_row

root = Path(__file__).parent.parent
raw = root / "data" / "raw"
contrib = raw / "contrib"
table = raw / "metadata_contrib.csv"
source = Path(sys.argv[1])

contrib.mkdir(parents=True, exist_ok=True)
old = pd.read_csv(table, dtype=str) if table.exists() else pd.DataFrame(columns=["id"])
rows, added = old.to_dict("records"), 0
for folder in sorted(p for p in source.iterdir() if (p / "metadata.json").exists()):
    record = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    out = contrib / f"{record['submission_id']}.txt"
    row = library_row(record, f"contrib/{out.name}")
    if row["id"] in set(old["id"]):
        continue
    shutil.copy(folder / "spectrum.txt", out)  # только спектр: исходник и e-mail остаются закрытыми
    rows.append(row)
    added += 1
    print(f"+ {row['id']}: {row['material']}, {row['laser_nm']:.0f} нм ({row['source']})")

pd.DataFrame(rows).to_csv(table, index=False)
n = build_library(metadata_csv=table, raw_dir=raw, out_dir=root / "data" / "library_contrib")
print(f"Добавлено {added}; в базе коллег {n} эталонов → data/library_contrib/")
