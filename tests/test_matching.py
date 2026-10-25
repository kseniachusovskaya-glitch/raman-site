"""Тесты для ramancore/matching.py."""

import numpy as np
import pytest

from ramancore.library import build_library, load_library
from ramancore.matching import hqi, search

from conftest import SYNTHETIC, make_spectrum


@pytest.fixture
def library(library_files):
    meta_csv, raw, out = library_files
    build_library(meta_csv, raw, out)
    return load_library(out)


def test_hqi_basic():
    a = np.array([0.0, 1.0, 2.0, 1.0])
    assert hqi(a, a) == pytest.approx(100)
    assert hqi(a, 3 * a) == pytest.approx(100)  # от масштаба не зависит
    assert hqi([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0)


@pytest.mark.parametrize("material", list(SYNTHETIC))
def test_noisy_reference_found_first(library, material):
    # Критерий этапа 1: зашумлённый эталон из базы — на 1-м месте с HQI > 90.
    # Шум σ = 10 при пике 1000 (отношение сигнал/шум 100).
    # Ограничение: при σ = 20 HQI падает до 70–90 — первое место сохраняется,
    # но порог 90 не всегда проходит.
    x, y = make_spectrum(SYNTHETIC[material], noise=10, seed=100)
    result = search(x, y, library)
    assert result.loc[0, "material"] == material
    assert result.loc[0, "hqi"] > 90


def test_different_instrument_step_still_found(library):
    # Тот же GaAs, но с шагом 6.24 см⁻¹, как на вашем приборе, и пики шире
    peaks = [(268, 10, 1000), (292, 10, 600)]
    x, y = make_spectrum(peaks, step=6.24, noise=10, seed=7)
    result = search(x, y, library)
    assert result.loc[0, "material"] == "GaAs"


def test_result_table(library):
    x, y = make_spectrum(SYNTHETIC["Si"], seed=5)
    result = search(x, y, library, top=3)
    assert len(result) == 3
    assert list(result.columns) == ["id", "material", "laser_nm", "hqi", "hqi_deriv", "source", "license"]
    assert result["hqi"].is_monotonic_decreasing  # по убыванию
    assert result["hqi"].between(0, 100).all() and result["hqi_deriv"].between(0, 100).all()


@pytest.mark.parametrize("material", list(SYNTHETIC))
def test_very_noisy_reference_still_first(library, material):
    x, y = make_spectrum(SYNTHETIC[material], noise=20, seed=100)
    assert search(x, y, library).loc[0, "material"] == material


def test_class_filter(library):
    lib = dict(library)
    meta = lib["metadata"].copy()
    meta.loc[meta["material"] == "Si", "class"] = "mineral"  # пусть Si будет «минералом»
    lib["metadata"] = meta
    x, y = make_spectrum(SYNTHETIC["Si"], seed=5)
    assert search(x, y, lib).loc[0, "material"] == "Si"
    only_semi = search(x, y, lib, classes=["semiconductor"])
    assert "Si" not in set(only_semi["material"])
    assert list(only_semi.index) == list(range(len(only_semi)))
