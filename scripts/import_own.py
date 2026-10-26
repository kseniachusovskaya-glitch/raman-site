"""Готовит свои спектры к базе: усредняет повторы и пишет metadata_own.csv.

Описание образцов — в samples.csv в той же папке, что и спектры:
одна строка = один эталон.
Столбец pattern — шаблон имён файлов внутри папки со спектрами
(* — любые символы, ? — один символ). Если под шаблон подходит несколько
файлов (повторы 1, 2, 3), они усредняются в один спектр.

Запуск (путь — папка, где лежат спектры):
    uv run python scripts/import_own.py "data/examples/16.03.26 Спектры для базы"

Результат:
    data/raw/own/<id>.txt       — усреднённые спектры (x и y через табуляцию);
    data/raw/metadata_own.csv   — их описание для scripts/build_library.py.
Обе вещи в .gitignore: свои данные не попадают в открытый репозиторий.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ramancore.io import read_spectrum
from ramancore.preprocess import average_spectra

root = Path(__file__).parent.parent
own = root / "data" / "raw" / "own"
source = Path(sys.argv[1])

own.mkdir(parents=True, exist_ok=True)
samples = pd.read_csv(source / "samples.csv", dtype=str).fillna("")
rows = []
for _, s in samples.iterrows():
    files = sorted(source.glob(s["pattern"]))
    if not files:
        sys.exit(f"{s['id']}: нет файлов по шаблону {s['pattern']!r}")
    x, y = average_spectra([read_spectrum(f)[:2] for f in files])  # [:2] — только x и y
    out = own / f"{s['id']}.txt"
    np.savetxt(out, np.column_stack([x, y]), fmt="%.3f", delimiter="\t",
               header=f"{s['id']}: среднее {len(files)} спектров", encoding="utf-8")
    row = s.drop("pattern").to_dict()
    row["file"] = f"own/{out.name}"
    names = "; ".join(f.relative_to(source).as_posix() for f in files)
    row["note"] = f"{row['note']}; среднее {len(files)} спектров: {names}".lstrip("; ")
    rows.append(row)
    print(f"{s['id']}: {len(files)} файл(ов)")

pd.DataFrame(rows).to_csv(root / "data" / "raw" / "metadata_own.csv", index=False)
print(f"Готово: {len(rows)} эталонов → data/raw/metadata_own.csv")
