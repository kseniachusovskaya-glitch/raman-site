"""База эталонов: сборка из исходных файлов и загрузка.

Как устроена база (папка data/library/):
  metadata.csv — по строке на эталон: откуда он, условия съёмки, лицензия;
  spectra.npz  — общая сетка x и матрица обработанных спектров
                 (строка матрицы = эталон, в том же порядке, что metadata.csv),
                 плюс параметры конвейера, которыми их обработали.

Исходные файлы лежат в data/raw/ как скачаны. Базу всегда можно
пересобрать при смене параметров обработки (scripts/build_library.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ramancore.io import read_spectrum
from ramancore.preprocess import process

# Обязательные поля metadata.csv (схема из плана проекта) + file — путь к
# исходному файлу относительно папки raw
REQUIRED = ["id", "file", "material", "formula", "class", "laser_nm", "source", "license"]
OPTIONAL = ["phase_or_polytype", "power_mW", "instrument", "temperature_K", "note"]

# Параметры конвейера для базы. Пользовательский спектр при поиске
# обрабатывается ТЕМИ ЖЕ параметрами — они сохраняются в spectra.npz.
# Для поиска выбраны arpls и сглаживание 5 см⁻¹ на общей сетке: на
# синтетике с шумом 1 % от пика HQI своего эталона 95–99 против 87–98 с asls
# без сглаживания (asls оставляет под шумом смещение ~2σ, которое на 3000
# точках заметно снижает HQI).
DEFAULT_PARAMS = {"xmin": 100, "xmax": 3200, "baseline_method": "arpls", "lam": 1e5,
                  "smooth_window": 5, "norm": "vector", "step": 1.0}


def _check_metadata(meta):
    """Проверяет metadata.csv: без источника и лицензии эталон публиковать нельзя."""
    missing = [c for c in REQUIRED if c not in meta.columns]
    if missing:
        raise ValueError(f"В metadata.csv нет обязательных столбцов: {missing}")
    for col in REQUIRED:
        empty = meta[col].isna() | (meta[col].astype(str).str.strip() == "")
        if empty.any():
            ids = list(meta.loc[empty, "id"])
            raise ValueError(f"Пустое обязательное поле {col!r} у эталонов: {ids}")
    if meta["id"].duplicated().any():
        raise ValueError(f"Повторяются id: {list(meta.loc[meta['id'].duplicated(), 'id'])}")


def build_library(metadata_csv, raw_dir, out_dir, params=None):
    """Собирает базу: читает каждый файл из metadata_csv, обрабатывает
    конвейером process и сохраняет metadata.csv + spectra.npz в out_dir.

    params — параметры process; по умолчанию DEFAULT_PARAMS.
    Возвращает число эталонов в базе.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}  # ** — объединение словарей
    meta = pd.read_csv(metadata_csv)
    _check_metadata(meta)

    rows = []
    grid = None
    for _, ref in meta.iterrows():
        x, y, m = read_spectrum(Path(raw_dir) / ref["file"], laser_nm=ref["laser_nm"])
        grid, y_processed, _ = process(x, y, m, **params)
        rows.append(y_processed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    np.savez_compressed(
        out_dir / "spectra.npz",
        grid=grid,
        spectra=np.vstack(rows),  # vstack — складывает строки в матрицу
        ids=meta["id"].to_numpy(dtype=str),
        params=json.dumps(params),
    )
    return len(meta)


def load_library(library_dir="data/library"):
    """Загружает базу. Возвращает словарь:
      grid     — общая сетка x;
      spectra  — матрица (эталоны × точки сетки);
      metadata — pandas.DataFrame с описанием эталонов;
      params   — параметры конвейера, которыми обработаны эталоны.
    """
    library_dir = Path(library_dir)
    data = np.load(library_dir / "spectra.npz")
    metadata = pd.read_csv(library_dir / "metadata.csv")
    if list(data["ids"]) != list(metadata["id"].astype(str)):
        raise ValueError("spectra.npz и metadata.csv не совпадают — пересоберите базу")
    return {
        "grid": data["grid"],
        "spectra": data["spectra"],
        "metadata": metadata,
        "params": json.loads(str(data["params"])),
    }
