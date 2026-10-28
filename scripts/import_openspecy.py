"""Добавляет в базу рамановские эталоны из библиотеки Open Specy.

Как пользоваться:
  1. Скачайте библиотеку с https://osf.io/x7dpz/ (Download As Zip) и
     достаньте из архива файл raw.rds.
  2. Читалка формата R (пакет rdata) ставится вместе с uv sync (группа dev).
  3. Запустите из папки проекта raman-site:
         uv run python scripts/import_openspecy.py "<путь>/raw.rds"
  4. Закоммитьте data/raw/metadata_openspecy.csv и data/library_openspecy/ —
     они открытые (CC BY 4.0). Сами спектры пишутся в data/raw/openspecy/
     (не в git: их всегда можно получить заново этим скриптом).

Файл .rds читается пакетом rdata — он только разбирает данные и не
выполняет никакой код из файла (в отличие от pickle).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import rdata

from ramancore.library import build_library
from ramancore.openspecy import library_row, select_raman, to_table, usable

root = Path(__file__).parent.parent
raw = root / "data" / "raw"
out_txt = raw / "openspecy"
table = raw / "metadata_openspecy.csv"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # rdata предупреждает о классе data.table — это не важно
    lib = rdata.read_rds(sys.argv[1])
wavenumber = np.asarray(lib["wavenumber"], dtype=float)
spectra = lib["spectra"]          # столбец = спектр, строки — точки wavenumber
metadata = lib["metadata"].reset_index(drop=True)

chosen = select_raman(metadata)
out_txt.mkdir(parents=True, exist_ok=True)
rows, skipped = [], 0
for i, row in chosen.iterrows():
    spectrum = usable(wavenumber, spectra.iloc[:, i].to_numpy(dtype=float))
    if spectrum is None:
        skipped += 1
        continue
    file = f"openspecy/{row['sample_name']}.txt"
    np.savetxt(raw / file, np.column_stack(spectrum), fmt="%.4f", delimiter="\t")
    rows.append(library_row(row, file))

to_table(rows).to_csv(table, index=False)
n = build_library(metadata_csv=table, raw_dir=raw, out_dir=root / "data" / "library_openspecy")
print(f"Отобрано {len(chosen)}, пропущено (мало точек) {skipped}; "
      f"в базе Open Specy {n} эталонов → data/library_openspecy/")
