"""Тесты для ramancore/carbon.py на синтетических спектрах DLC."""

import numpy as np
import pytest

from ramancore.carbon import bwf, bwf_maximum, fit_dlc, g_dispersion, hydrogen_content
from ramancore.peaks import lorentzian


def dlc(d_amp=400, g_amp=1000, g_c=1560, q=-5, slope=0.5, noise=5, seed=0):
    """Синтетика: D (лоренциан) + G (BWF) + наклонный фон + шум."""
    x = np.arange(600, 2200, 1.21)
    y = lorentzian(x, 1350, 250, d_amp) + bwf(x, g_c, 150, g_amp, q) + 300 + slope * (x - 1000)
    return x, y + np.random.default_rng(seed).normal(0, noise, x.size)


def test_bwf_maximum_numerically():
    x = np.arange(1300, 1800, 0.01)
    y = bwf(x, 1560, 150, 1000, -5)
    pos, k = bwf_maximum(1560, 150, -5)
    assert x[np.argmax(y)] == pytest.approx(pos, abs=0.02)  # 1560 − 15 = 1545
    assert y.max() == pytest.approx(1000 * k, rel=1e-6)


def test_bwf_becomes_lorentzian():
    x = np.linspace(1400, 1700, 50)
    assert np.allclose(bwf(x, 1560, 150, 1, -1e9), lorentzian(x, 1560, 150, 1))


def test_fit_recovers_parameters():
    x, y = dlc()
    r = fit_dlc(x, y)
    pos, k = bwf_maximum(1560, 150, -5)
    assert r["pos_g"] == pytest.approx(pos, abs=1)
    assert r["fwhm_g"] == pytest.approx(150, abs=3)
    assert r["id_ig"] == pytest.approx(400 / (1000 * k), abs=0.02)
    # m/I(G) в мкм: наклон 0.5 на см⁻¹ / высота G
    assert r["m_ig"] == pytest.approx(0.5 / (1000 * k) * 1e4, rel=0.05)


def test_no_d_band_warns():
    x, y = dlc(d_amp=0)
    r = fit_dlc(x, y)
    assert r["id_ig"] < 0.03
    assert any("D почти не видна" in w for w in r["warnings"])


def test_hydrogen_formula():
    # Casiraghi et al. 2005: 1 мкм → 21.7 %, 10 мкм → 38.3 %
    assert hydrogen_content(1.0)[0] == pytest.approx(21.7)
    assert hydrogen_content(10.0)[0] == pytest.approx(38.3)
    h, warnings = hydrogen_content(-0.1)
    assert np.isnan(h) and warnings
    assert any("514" in w for w in hydrogen_content(5.0, laser_nm=785)[1])
    assert not any("514" in w for w in hydrogen_content(5.0, laser_nm=532)[1])


def test_g_dispersion():
    assert g_dispersion(1560, 532, 1545, 785) == pytest.approx(15 / 253)
    with pytest.raises(ValueError):
        g_dispersion(1560, 532, 1550, 532)
