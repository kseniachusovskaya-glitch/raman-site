"""Тесты для ramancore/polymers.py на синтетических спектрах ПВДФ."""

import numpy as np
import pytest

from ramancore.peaks import gaussian
from ramancore.polymers import fit_pvdf, fit_pvdf_amorphous, phase_fractions, pvdf_cf2_ratio


def _lor(x, c, w, a):
    return a / (1 + ((x - c) / (w / 2)) ** 2)


def _pvdf(alpha, gamma, beta, tfe=0.0, noise=2.0, seed=0, w=10.0):
    """Спектр 700–950 см⁻¹: полосы α, γ, β одной ширины с высотами alpha, gamma,
    beta + νs(CC) 881 + наклонный фон и шум."""
    x = np.arange(700.0, 950.0, 1.0)
    y = 50 + 0.05 * x + _lor(x, 795, w, alpha) + _lor(x, 811, w, gamma) \
        + _lor(x, 839, w, beta) + _lor(x, 881, 18, 300) + _lor(x, 827, 8, tfe)
    return x, y + np.random.default_rng(seed).normal(0, noise, x.size)


def test_fractions_from_heights():
    # Деримедведь 2025, ориентированная плёнка: α:γ:β = 19:26:55
    x, y = _pvdf(190, 260, 550)
    r = fit_pvdf(x, y, shape="lorentz")
    f = r["fractions"]["по площади, %"]
    assert f["α"] == pytest.approx(19, abs=1.5)
    assert f["γ"] == pytest.approx(26, abs=1.5)
    assert f["β"] == pytest.approx(55, abs=1.5)
    assert r["fractions"]["по площади, %"].sum() == pytest.approx(100)
    assert r["table"].loc["β", "center"] == pytest.approx(839, abs=0.3)
    assert r["beta_alpha"] == pytest.approx(550 / 190, rel=0.05)


def test_alpha_only_film():
    # Неориентированная α-плёнка: β почти нет
    x, y = _pvdf(1000, 0, 30)
    r = fit_pvdf(x, y, gamma=False)
    assert r["fractions"]["по площади, %"]["α"] > 90
    assert "γ" not in r["fractions"].index
    assert r["beta_alpha"] < 0.1


def test_tfe_band_does_not_inflate_beta():
    x, y = _pvdf(240, 390, 370, tfe=80)
    with_tfe = fit_pvdf(x, y, tfe=True, shape="lorentz")["fractions"]["по площади, %"]
    assert with_tfe["β"] == pytest.approx(37, abs=2.5)
    assert "ТФЭ" not in with_tfe.index  # ТФЭ — не фаза ПВДФ


def test_short_spectrum_rejected():
    x = np.arange(500.0, 700.0)
    with pytest.raises(ValueError):
        fit_pvdf(x, np.ones_like(x))


def test_cf2_ratio():
    x = np.arange(400.0, 700.0)
    y = 20 + _lor(x, 511, 9, 200) + _lor(x, 610, 9, 50) \
        + np.random.default_rng(1).normal(0, 1, x.size)
    ratio, table = pvdf_cf2_ratio(x, y, shape="lorentz")
    assert ratio == pytest.approx(4.0, rel=0.08)
    assert list(table.index) == ["β (CF2, 511)", "α (CF2, 610)"]


def test_phase_fractions_negative_clipped():
    import pandas as pd
    t = pd.DataFrame({"area": [10.0, -1.0, 30.0], "amplitude": [1.0, 0.0, 3.0]},
                     index=["α", "γ", "β"])
    f = phase_fractions(t, ["α", "γ", "β"])
    assert f.loc["γ", "по площади, %"] == 0
    assert f.loc["β", "по площади, %"] == pytest.approx(75)


def test_amorphous_model_recovers_beta_fraction():
    # Как образец E у Chapron 2021 (λ = 4): Fβ ≈ 70 % в кристалле, аморфной ~40 %
    x = np.arange(740.0, 910.0, 1.0)
    w_c = 8.0
    # площади лоренцианов: α 795 + 874 — 30 %, β 839 — 70 %
    a795, a874, a839 = 200.0, 100.0, 700.0
    amp = lambda area, w: area / (np.pi * w / 2)  # высота лоренциана по площади
    y = 20 + 0.02 * x
    y = y + _lor(x, 795, w_c, amp(a795, w_c)) + _lor(x, 874, w_c, amp(a874, w_c)) \
        + _lor(x, 839, w_c, amp(a839, w_c))
    for c, h in [(797, 8), (809, 6), (830, 6), (881, 8)]:
        y = y + gaussian(x, c, 30, h)
    y = y + np.random.default_rng(3).normal(0, 0.3, x.size)
    r = fit_pvdf_amorphous(x, y)
    assert r["f_beta"] == pytest.approx(70, abs=4)
    assert r["f_alpha"] + r["f_beta"] == pytest.approx(100)
    assert 0 < r["f_amorphous"] < 100 and r["fit"]["r2"] > 0.99
