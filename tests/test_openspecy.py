"""Тесты для ramancore/openspecy.py на маленькой выдуманной таблице."""

import numpy as np
import pandas as pd
import pytest

from ramancore.library import build_library, load_library
from ramancore.openspecy import (formula_from_rod, laser_from_text, library_row, select_raman,
                                 to_table, usable)


@pytest.mark.parametrize("texts, expected", [
    (("532 nm",), 532.0),
    (("785",), 785.0),
    (("532 1/cm",), 532.0),
    ((None, "alizarin 23600_532"), 532.0),   # номер 23600 не принимается за лазер
    ((np.nan, "polystyrene"), np.nan),
])
def test_laser_from_text(texts, expected):
    got = laser_from_text(*texts)
    assert (np.isnan(got) and np.isnan(expected)) or got == expected


def test_formula_from_rod():
    assert formula_from_rod("fe3 o4") == "Fe3O4"
    assert formula_from_rod("f4 li y") == "F4LiY"


def _metadata():
    return pd.DataFrame({
        "sample_name": ["a1", "b2", "c3", "d4", "e5"],
        "spectrum_type": ["raman", "raman", "ftir", "raman", "raman"],
        "organization": ["slopp, munno et al. 2020", "rruff", "nist", "raman open database",
                         "microplastic solution"],
        "spectrum_identity": ["polyacrylonitrile", "quartz", "pe", "fe3 o4", "pvdf"],
        "plastic_or_not": ["plastic", "not plastic", "plastic", "not plastic", "plastic"],
        "material_class": ["polyacrylonitriles", "mineral", "polyolefins", "mineral", "x"],
        "laser_light_used": ["785", None, None, None, None],
        "license": [None, None, None, None, "cc by-nc"],
        "url": [None, "rruff.info/r1", None, None, None],
    })


def test_select_raman_skips_rruff_ftir_and_nc():
    chosen = select_raman(_metadata())
    assert list(chosen["sample_name"]) == ["a1", "d4"]


def test_library_row():
    m = select_raman(_metadata())
    rows = [library_row(r, f"openspecy/{r['sample_name']}.txt") for _, r in m.iterrows()]
    polymer, rod = rows
    assert polymer["class"] == "polymer" and polymer["material"] == "polyacrylonitrile"
    assert polymer["laser_nm"] == 785.0 and polymer["formula"] == "н/д"
    assert "slopp" in polymer["source"] and "CC BY 4.0" in polymer["license"]
    assert rod["material"] == rod["formula"] == "Fe3O4" and rod["class"] == "mineral"
    assert np.isnan(rod["laser_nm"]) and rod["url"] == "https://osf.io/x7dpz/"


def test_usable():
    x = np.arange(102.0, 3000.0, 6.0)
    y = np.full(x.size, np.nan)
    y[10:200] = np.sin(x[10:200] / 50)  # измерено 190 точек
    xs, ys = usable(x, y)
    assert len(xs) == 190 and np.all(np.isfinite(ys))
    y[:] = np.nan
    y[10:30] = 1.0 + np.arange(20)  # 20 точек — мало
    assert usable(x, y) is None


def test_rows_build_library_without_laser(tmp_path):
    # Эталон без длины волны лазера тоже собирается в базу и ищется
    raw = tmp_path / "raw"
    (raw / "openspecy").mkdir(parents=True)
    x = np.arange(102.0, 3200.0, 6.0)
    rows = []
    for i, (name, center) in enumerate([("pe", 1296.0), ("ps", 1002.0)]):
        y = 10 + 1000 / (1 + ((x - center) / 6) ** 2)
        np.savetxt(raw / "openspecy" / f"{name}.txt", np.column_stack([x, y]), delimiter="\t")
        meta = {"sample_name": name, "organization": "nist", "spectrum_identity": name,
                "plastic_or_not": "plastic"}
        rows.append(library_row(meta, f"openspecy/{name}.txt"))
    table = tmp_path / "metadata_openspecy.csv"
    to_table(rows).to_csv(table, index=False)
    assert build_library(table, raw, tmp_path / "library") == 2
    library = load_library(tmp_path / "library")
    assert library["metadata"]["laser_nm"].isna().all()
