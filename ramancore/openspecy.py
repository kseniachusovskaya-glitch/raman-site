"""Эталоны из открытой библиотеки Open Specy (полимеры, органика, пигменты).

Open Specy (Cowger et al., Anal. Chem. 2021, doi 10.1021/acs.analchem.1c00123;
библиотека — osf.io/x7dpz) распространяется по лицензии CC BY 4.0
(файл LICENSE.md в github.com/wincowgerDEV/OpenSpecy-package). Библиотека
собрана из многих источников (SLoPP, Raman Open Database, Horiba, NIST, …),
поэтому у каждого эталона указываются и Open Specy, и исходный источник.

Здесь только отбор и перевод описаний в схему metadata.csv нашей базы;
чтение файла .rds — в scripts/import_openspecy.py.
"""

import re

import numpy as np
import pandas as pd

LICENSE = "CC BY 4.0 (Open Specy: cite Cowger et al. 2021)"
# RRUFF уже есть в основной базе целиком — второй раз не берём
SKIP_ORGANIZATIONS = {"rruff"}
MIN_POINTS = 50       # меньше точек — не спектр для поиска
MIN_RANGE_CM = 300.0  # и диапазон не уже 300 см⁻¹ (как в форме для коллег — 200, тут строже)


def laser_from_text(*texts):
    """Длина волны лазера из подписи: «532 nm», «785», «alizarin 23600_532» → 532.0.
    Берётся число 400–1100 (так не путается с номером образца); нет — nan."""
    for text in texts:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            continue
        for number in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", str(text)):
            if 400 <= int(number) <= 1100:
                return float(number)
    return np.nan


def formula_from_rod(identity):
    """В Raman Open Database вместо названия записана формула строчными
    буквами: «fe3 o4» → «Fe3O4», «f4 li y» → «F4LiY»."""
    return "".join(part[:1].upper() + part[1:] for part in str(identity).split())


def material_class(row):
    """Класс нашей базы: polymer (пластики), mineral или other."""
    if str(row.get("plastic_or_not", "")).lower() == "plastic":
        return "polymer"
    if str(row.get("material_class", "")).lower() == "mineral":
        return "mineral"
    return "other"


def select_raman(metadata):
    """Отбирает рамановские спектры, которые можно взять в открытую базу:
    не RRUFF (он уже есть) и без пометки «некоммерческая» (NC) в license."""
    m = metadata
    raman = m["spectrum_type"].astype(str).str.lower() == "raman"
    other = ~m["organization"].astype(str).str.lower().isin(SKIP_ORGANIZATIONS)
    license_text = m["license"].astype(str).str.lower() if "license" in m else ""
    no_nc = ~license_text.str.contains("nc", regex=False) if "license" in m else True
    return m[raman & other & no_nc]


def library_row(row, file):
    """Строка metadata.csv нашей базы из описания эталона Open Specy."""
    org = str(row["organization"])
    identity = str(row["spectrum_identity"]).strip().rstrip(";").strip()
    is_rod = org.lower() == "raman open database"
    formula = formula_from_rod(identity) if is_rod else "н/д"
    note = [f"исходный источник: {org}"]
    for key in ("citation", "instrument_used", "material_form"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
            note.append(f"{key}: {value.strip()}")
    url = row.get("url")
    return {
        "id": f"openspecy_{row['sample_name']}",
        "file": file,
        "material": formula if is_rod else identity,
        "formula": formula,
        "class": material_class(row),
        "laser_nm": laser_from_text(row.get("laser_light_used"), identity),
        "source": f"Open Specy / {org}",
        "license": LICENSE,
        "url": url if isinstance(url, str) and url else "https://osf.io/x7dpz/",
        "note": "; ".join(note),
    }


def usable(x, y):
    """Спектр из общей матрицы Open Specy (вне диапазона измерения там nan):
    возвращает (x, y) только по измеренным точкам или None, если для
    базы мало точек или узкий диапазон."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < MIN_POINTS or np.ptp(x[ok]) < MIN_RANGE_CM or np.ptp(y[ok]) == 0:
        return None
    return x[ok], y[ok]


def to_table(rows):
    """Список строк library_row → DataFrame со столбцами в порядке схемы."""
    columns = ["id", "file", "material", "formula", "class", "laser_nm", "source", "license",
               "url", "note"]
    return pd.DataFrame(rows).reindex(columns=columns)
