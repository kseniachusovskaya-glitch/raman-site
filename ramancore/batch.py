"""Пакетная обработка: много спектров за раз (серии, карты, повторы).

Спектры серии — список троек (name, x, y): имя файла и сам спектр.
Каждый спектр обрабатывается теми же функциями ядра, что и одиночный
(fit_peaks, search), а результаты собираются в одну таблицу.

Число в имени файла (например, «T300K», «x_12,5», «point_07») берётся
как подпись точки серии: по ней строится график «параметр — подпись».
"""

import re

import numpy as np
import pandas as pd

from ramancore.matching import search
from ramancore.peaks import fit_peaks

PEAK_COLUMNS = ["center", "center_err", "fwhm", "fwhm_err", "amplitude", "amplitude_err",
                "area", "area_err"]

# Число: необязательный минус, цифры, необязательная дробная часть через точку или запятую
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def number_from_name(name):
    """Последнее число в имени файла без расширения: «T300K.txt» → 300.0,
    «map_x12,5.txt» → 12.5. Нет числа — None.

    Минус считается частью числа, только если перед ним не буква и не цифра
    («x-5» → 5, а «_-5» → −5): в именах дефис чаще разделитель.
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,4}$", "", str(name))  # убираем расширение
    matches = list(_NUMBER.finditer(stem))
    if not matches:
        return None
    m = matches[-1]
    text = m.group()
    if text.startswith("-") and m.start() > 0 and stem[m.start() - 1].isalnum():
        text = text[1:]
    return float(text.replace(",", "."))


def sort_series(spectra):
    """Упорядочивает серию по числу в имени (файлы без числа — в конце, по имени)."""
    def key(item):
        number = number_from_name(item[0])
        return (number is None, number if number is not None else 0.0, item[0])
    return sorted(spectra, key=key)


def fit_series(spectra, centers, shape="lorentz", margin=30.0, track=False):
    """Подгоняет одни и те же пики во всех спектрах серии.

    centers — примерные положения пиков (одинаковые для всех файлов).
              fit_peaks уточняет центр, но не дальше примерно одной ширины
              пика от начального значения.
    track   — следить за сдвигом: каждый следующий файл начинается с
              центров, найденных в предыдущем. Нужно, когда пик за серию
              уходит дальше своей ширины (нагрев, карта напряжений).
              Серию тогда стоит упорядочить (sort_series).
    Если подгонка какого-то файла не сошлась, он не прерывает серию:
    в таблице у него будет текст ошибки в столбце error. Если центр
    упёрся в границу, об этом пишется в столбце warning.

    Возвращает «длинную» таблицу: одна строка — один пик одного файла.
    Столбцы: file, label (число из имени), peak (1, 2, … по возрастанию
    центра), center, fwhm, amplitude, area с погрешностями *_err, r2,
    error, warning.
    """
    centers = np.sort(np.atleast_1d(np.asarray(centers, dtype=float)))
    start = centers
    rows = []
    for name, x, y in spectra:
        base = {"file": name, "label": number_from_name(name)}
        try:
            table, fit = fit_peaks(x, y, start, shape=shape, margin=margin)
        except (ValueError, RuntimeError) as error:
            for i in range(len(centers)):
                rows.append({**base, "peak": i + 1, "error": str(error), "warning": ""})
            continue
        for i, r in table.reset_index(drop=True).iterrows():
            # fit_peaks пускает центр только в пределах center_bounds;
            # если центр у самой границы, настоящий пик, скорее всего, дальше
            lo, hi = fit["center_bounds"][i]
            edge = 0.02 * (hi - lo)
            warning = ("центр у границы подгонки: пик сдвинут сильнее своей ширины"
                       if r["center"] - lo < edge or hi - r["center"] < edge else "")
            rows.append({**base, "peak": i + 1, **{c: r[c] for c in PEAK_COLUMNS},
                         "r2": fit["r2"], "error": "", "warning": warning})
        if track:
            start = np.sort(table["center"].to_numpy())
    columns = ["file", "label", "peak"] + PEAK_COLUMNS + ["r2", "error", "warning"]
    # reindex добавляет недостающие столбцы (у файлов с ошибкой — пустые)
    return pd.DataFrame(rows).reindex(columns=columns)


def wide_table(long):
    """Из длинной таблицы fit_series — одна строка на файл:
    center_1, fwhm_1, area_1, …, center_2, … (номер — номер пика)."""
    keep = ["center", "center_err", "fwhm", "fwhm_err", "area", "area_err"]
    # pivot «разворачивает» строки пиков в столбцы
    wide = long.pivot_table(index=["file"], columns="peak", values=keep, aggfunc="first",
                            dropna=False)
    wide.columns = [f"{value}_{peak}" for value, peak in wide.columns]
    order = [f"{v}_{p}" for p in sorted(long["peak"].unique()) for v in keep]
    info = long.groupby("file", sort=False).agg(
        label=("label", "first"), r2=("r2", "first"), error=("error", "first"),
        # предупреждения всех пиков файла — через «; »
        warning=("warning", lambda w: "; ".join(sorted({v for v in w if v}))))
    wide = info.join(wide[[c for c in order if c in wide.columns]])
    # порядок файлов — как в исходной серии
    return wide.loc[long["file"].drop_duplicates()].reset_index()


def peak_ratio(long, numerator, denominator, value="area"):
    """Отношение двух пиков в каждом файле, например I(LO)/I(TO) или I_D/I_G.

    numerator, denominator — номера пиков (1, 2, …); value — "area" или
    "amplitude". Погрешность — по правилу для частного:
        δR/R = √((δa/a)² + (δb/b)²).
    Возвращает таблицу: file, label, ratio, ratio_err.
    """
    if value not in ("area", "amplitude"):
        raise ValueError('value должно быть "area" или "amplitude"')
    a = long[long["peak"] == numerator].set_index("file")
    b = long[long["peak"] == denominator].set_index("file")
    ratio = a[value] / b[value]
    rel = np.sqrt((a[f"{value}_err"] / a[value]) ** 2 + (b[f"{value}_err"] / b[value]) ** 2)
    out = pd.DataFrame({"label": a["label"], "ratio": ratio, "ratio_err": np.abs(ratio) * rel})
    return out.reset_index()


def search_series(spectra, library, top=1, classes=None):
    """Ищет по базе каждый спектр серии. Возвращает таблицу:
    file, rank (1 — лучшее совпадение), id, material, hqi, hqi_deriv, …
    (top строк на файл)."""
    parts = []
    for name, x, y in spectra:
        result = search(x, y, library, top=top, classes=classes)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        result.insert(0, "file", name)
        parts.append(result)
    if not parts:
        return pd.DataFrame(columns=["file", "rank"])
    return pd.concat(parts, ignore_index=True)
