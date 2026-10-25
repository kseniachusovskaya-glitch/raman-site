"""Поиск похожих эталонов в базе по мере сходства HQI.

HQI = 100 · (u·r)² / ((u·u)(r·r)), где u — спектр пользователя, r — эталон.
Это квадрат косинуса угла между спектрами как векторами: 100 — формы
совпадают, 0 — ничего общего. Спектры в базе уже нормированы (vector),
поэтому для всей базы это одно матричное умножение.
"""

import numpy as np
import pandas as pd

from ramancore.preprocess import process


def hqi(a, b):
    """HQI двух спектров на одной сетке (0–100)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return 100 * np.dot(a, b) ** 2 / (np.dot(a, a) * np.dot(b, b))


def _unit_rows(matrix):
    """Нормирует каждую строку матрицы на длину 1."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def search(x, y, library, top=10, meta=None):
    """Ищет эталоны, похожие на спектр (x, y).

    Спектр сначала проходит тот же конвейер, что и эталоны
    (параметры берутся из library["params"]).

    Считаются две меры:
      hqi       — по самим спектрам;
      hqi_deriv — по первым производным: меньше зависит от остатков фона.

    Возвращает pandas.DataFrame из top лучших эталонов, отсортированный
    по hqi: id, material, phase_or_polytype, hqi, hqi_deriv, source, license.
    """
    grid, u, _ = process(x, y, meta, **library["params"])
    if not np.array_equal(grid, library["grid"]):
        raise ValueError("Сетка спектра не совпадает с сеткой базы")

    refs = library["spectra"]
    # @ — матричное умножение: скалярное произведение u с каждым эталоном сразу
    scores = 100 * (refs @ u) ** 2  # u и строки refs уже длины 1

    du = np.diff(u)
    du = du / np.linalg.norm(du)
    scores_deriv = 100 * (_unit_rows(np.diff(refs, axis=1)) @ du) ** 2

    table = library["metadata"].copy()
    table["hqi"] = scores
    table["hqi_deriv"] = scores_deriv
    columns = [c for c in ["id", "material", "phase_or_polytype", "hqi", "hqi_deriv",
                           "source", "license"] if c in table.columns]
    table = table.sort_values("hqi", ascending=False).head(top)
    return table[columns].reset_index(drop=True)
