"""Тесты для ramancore/batch.py на синтетических сериях."""

import numpy as np
import pytest

from ramancore.batch import (fit_series, number_from_name, peak_ratio, search_series,
                             sort_series, wide_table)
from ramancore.library import build_library, load_library

from conftest import SYNTHETIC, make_spectrum


@pytest.mark.parametrize("name, expected", [
    ("T300K.txt", 300.0),
    ("map_x12,5.txt", 12.5),
    ("point_07.csv", 7.0),
    ("Si-5.txt", 5.0),      # дефис после буквы — разделитель, не минус
    ("pos_-5.txt", -5.0),   # а после «_» — минус
    ("sample.txt", None),
    ("GaAs 2 run 3.txt", 3.0),  # берётся последнее число
])
def test_number_from_name(name, expected):
    assert number_from_name(name) == expected


def test_sort_series_by_number():
    items = [("b.txt", 0, 0), ("T10.txt", 0, 0), ("T2.txt", 0, 0), ("a.txt", 0, 0)]
    assert [n for n, *_ in sort_series(items)] == ["T2.txt", "T10.txt", "a.txt", "b.txt"]


def _si_series(shifts):
    """Серия Si: пик 520.7 + сдвиг, у каждого файла свой шум."""
    return [(f"pt_{i}.txt", *make_spectrum([(520.7 + s, 4, 1000)], noise=5, seed=i))
            for i, s in enumerate(shifts)]


def test_fit_series_small_shift():
    # Сдвиг меньше ширины пика (4 см⁻¹) — хватает одного центра для всех
    shifts = [0.0, -1.0, -2.0, 1.5]
    long = fit_series(_si_series(shifts), [520.7])
    assert len(long) == 4
    assert (long["error"] == "").all() and (long["warning"] == "").all()
    assert np.allclose(long["center"], 520.7 + np.array(shifts), atol=0.1)
    assert np.allclose(long["fwhm"], 4, atol=0.3)
    assert list(long["label"]) == [0, 1, 2, 3]


def test_fit_series_large_shift_needs_track():
    # Пик уходит на 10 см⁻¹ шагами по 2 — без слежения центр упирается в границу
    shifts = [0.0, -2.0, -4.0, -6.0, -8.0, -10.0]
    plain = fit_series(_si_series(shifts), [520.7])
    assert (plain["warning"] != "").any()
    tracked = fit_series(_si_series(shifts), [520.7], track=True)
    assert (tracked["warning"] == "").all()
    assert np.allclose(tracked["center"], 520.7 + np.array(shifts), atol=0.1)


def test_failed_file_does_not_stop_series():
    good = _si_series([0.0])
    bad = [("short.txt", np.array([519.0, 520.0, 521.0]), np.array([1.0, 2.0, 1.0]))]
    long = fit_series(good + bad, [520.7])
    assert long.loc[long["file"] == "pt_0.txt", "error"].item() == ""
    assert long.loc[long["file"] == "short.txt", "error"].item() != ""
    assert np.isnan(long.loc[long["file"] == "short.txt", "center"].item())


def test_two_peaks_ratio_and_wide():
    # GaAs: отношение LO/TO меняется по серии 0.5 → 2 (как при смене ориентации)
    ratios = [0.5, 1.0, 2.0]
    series = [(f"s{i}.txt", *make_spectrum([(268, 4, 1000), (292, 4, 1000 * r)], noise=3,
                                            seed=i)) for i, r in enumerate(ratios)]
    long = fit_series(series, [268, 292])
    table = peak_ratio(long, numerator=2, denominator=1)
    # Пики одной ширины и формы → отношение площадей = отношению высот
    assert np.allclose(table["ratio"], ratios, rtol=0.03)
    assert (table["ratio_err"] > 0).all() and (table["ratio_err"] < 0.05).all()
    amp = peak_ratio(long, 2, 1, value="amplitude")
    assert np.allclose(amp["ratio"], ratios, rtol=0.03)

    wide = wide_table(long)
    assert list(wide["file"]) == ["s0.txt", "s1.txt", "s2.txt"]
    assert {"center_1", "area_1", "center_2", "fwhm_2"} <= set(wide.columns)
    assert np.allclose(wide["center_2"], 292, atol=0.2)


def test_peak_ratio_bad_value():
    long = fit_series(_si_series([0.0]), [520.7])
    with pytest.raises(ValueError):
        peak_ratio(long, 1, 1, value="fwhm")


def test_search_series(library_files):
    meta_csv, raw, out = library_files
    build_library(meta_csv, raw, out)
    library = load_library(out)
    series = [(f"{m}.txt", *make_spectrum(SYNTHETIC[m], noise=10, seed=50 + i))
              for i, m in enumerate(["Si", "GaN", "MoS2"])]
    table = search_series(series, library, top=2)
    assert len(table) == 6
    best = table[table["rank"] == 1].set_index("file")["material"]
    assert best.to_dict() == {"Si.txt": "Si", "GaN.txt": "GaN", "MoS2.txt": "MoS2"}
