"""Выгрузка результатов в CSV.

В начале файла — строки-комментарии «# ключ: значение»: имя исходного
файла, лазер, история обработки и любые параметры. Так в одном файле
остаётся и результат, и то, как он получен. read_spectrum такие строки
пропускает, поэтому выгруженный спектр читается обратно без потерь.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _header_lines(meta, params):
    """Строки шапки: сначала meta (без history), потом params, потом history."""
    lines = []
    meta = dict(meta or {})  # dict(...) — копия, чтобы не менять словарь пользователя
    history = meta.pop("history", [])
    for key, value in {**meta, **(params or {})}.items():  # ** — слить два словаря
        if value is not None:
            lines.append(f"# {key}: {value}")
    for step in history:
        lines.append(f"# history: {step}")
    return lines


def _write(path, table, meta, params):
    """Пишет шапку-комментарий и таблицу в один файл."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:  # with — файл закроется сам
        for line in _header_lines(meta, params):
            f.write(line + "\n")
        # pandas пишет числа с полной точностью (как repr), поэтому
        # после чтения обратно получаются ровно те же значения
        table.to_csv(f, index=False, lineterminator="\n")
    return path


def to_csv(path, x, y, meta=None, params=None, extra=None):
    """Сохраняет спектр в CSV: столбцы raman_shift_cm-1 и intensity.

    meta   — словарь спектра из read_spectrum / process (лазер, история);
    params — параметры обработки или анализа, например {"baseline": "asls"};
    extra  — дополнительные столбцы той же длины, например
             {"baseline": baseline, "processed": y_processed}.

    Возвращает путь к файлу.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x и y разной длины")
    columns = {"raman_shift_cm-1": x, "intensity": y}
    for name, values in (extra or {}).items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(x):
            raise ValueError(f"Столбец {name!r}: длина {len(values)}, а у спектра {len(x)}")
        columns[name] = values
    return _write(path, pd.DataFrame(columns), meta, params)


def table_to_csv(path, table, meta=None, params=None):
    """Сохраняет таблицу (пики из fit_peaks, результат search) с той же шапкой.

    Прочитать обратно: pandas.read_csv(path, comment="#").
    """
    return _write(path, pd.DataFrame(table), meta, params)
