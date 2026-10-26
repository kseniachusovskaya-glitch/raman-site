"""Тесты для алмаза: напряжения (Ager & Drory 1993) и размер наноалмаза (Yoshikawa 1995)."""

import numpy as np
import pytest

from ramancore.carbon import confinement_spectrum, diamond_sp2_ratio, diamond_stress, nanodiamond_size
from ramancore.peaks import gaussian


def test_stress_coefficients():
    assert diamond_stress(1338.0, "singlet") == pytest.approx(-1.08 * 6)  # сжатие
    # Пример из статьи: синглет 1338 → 6.5 ГПа, дублет 1352 → 7.7 ГПа (по модулю)
    assert abs(diamond_stress(1338.0, "singlet")) == pytest.approx(6.5, abs=0.1)
    assert abs(diamond_stress(1352.0, "doublet")) == pytest.approx(7.7, abs=0.1)


def test_confinement_reproduces_paper():
    # Yoshikawa 1995: 55 Å → сдвиг −10 см⁻¹ и ширина 32 см⁻¹ от ω(0) = 1333
    w = np.arange(1150, 1400, 0.05)
    s = confinement_spectrum(w, 5.5)
    half = w[s >= 0.5]
    assert w[np.argmax(s)] - 1333 == pytest.approx(-10, abs=1)
    assert half.max() - half.min() == pytest.approx(32, abs=3)


def test_nanodiamond_size_recovered():
    x = np.arange(1150, 1450, 1.0)
    y = 500 * confinement_spectrum(x, 8.0) + 50 + 0.1 * (x - 1300)
    y = y + np.random.default_rng(0).normal(0, 3, x.size)
    r = nanodiamond_size(x, y)
    assert r["size"] == pytest.approx(8.0, rel=0.1)


def test_sp2_ratio():
    x = np.arange(1200, 1750, 1.0)
    # Гауссианы целиком помещаются в окна, поэтому площади считаются точно
    y = gaussian(x, 1332, 5, 1000) + gaussian(x, 1580, 50, 100) + 20
    ratio, sp2 = diamond_sp2_ratio(x, y)
    assert ratio == pytest.approx(50 * 100 / (5 * 1000), rel=0.05)  # площадь гауссиана ∝ w·h
    assert sp2 == pytest.approx(ratio / 60)


def test_split_line_found():
    from ramancore.carbon import fit_diamond_lines
    from ramancore.peaks import lorentzian
    x = np.arange(1250, 1450, 0.5)
    y = lorentzian(x, 1338, 6, 800) + lorentzian(x, 1352, 6, 500) + 30
    y = y + np.random.default_rng(0).normal(0, 3, x.size)
    table, _ = fit_diamond_lines(x, y)
    assert list(table.phonon) == ["singlet", "doublet"]
    assert table.center.tolist() == pytest.approx([1338, 1352], abs=0.5)
