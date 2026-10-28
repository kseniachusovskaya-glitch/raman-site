"""Тесты для ramancore/polymers.py на синтетических спектрах ПВДФ."""

import numpy as np
import pytest

from ramancore.peaks import gaussian
from ramancore.polymers import (MELT_AREA_RATIO, MELT_CENTERS, fit_pvdf, fit_pvdf_amorphous,
                                fit_pvdf_melt, melt_template, pearson7, pearson7_area,
                                phase_fractions, pvdf_cf2_ratio)


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


def test_pearson7_area_and_limits():
    x = np.arange(0.0, 2000.0, 0.01)
    assert pearson7_area(8, 5, 1.0) == pytest.approx(np.pi / 2 * 5 * 8)  # площадь лоренциана
    for m in (1.5, 3.0, 10.0):  # у m = 1 хвосты длинные — численно не сходится на 2000 см⁻¹
        y = pearson7(x, 1000, 8, 5, m)
        assert pearson7_area(8, 5, m) == pytest.approx(np.trapezoid(y, x), rel=1e-3)
        assert pearson7(1004.0, 1000, 8, 5, m) == pytest.approx(2.5)  # FWHM = 8
    assert np.allclose(pearson7(x[:5000], 20, 8, 5, 1.0), _lor(x[:5000], 20, 8, 5))


def _melt_amorphous(x, area=3000.0, fwhm=14.0):
    """Аморфный профиль как у Purushothaman 2024: 8 гауссианов, площади 8:4:4:2:1:2:4:1."""
    share = np.asarray(MELT_AREA_RATIO) / sum(MELT_AREA_RATIO)
    k = np.sqrt(np.pi / (4 * np.log(2)))
    return sum(gaussian(x, c, fwhm, area * s / (fwhm * k)) for c, s in zip(MELT_CENTERS, share))


def _crystal(x, areas):
    """Кристаллические полосы Pearson VII (m = 2, FWHM 6) с заданными площадями."""
    centers = {"α1": 799, "γ1": 814, "γ2": 830, "β": 842, "α2": 877, "c": 881}
    unit = pearson7_area(6, 1, 2)
    return sum(pearson7(x, centers[k], 6, a / unit, 2) for k, a in areas.items())


def test_melt_paper_ratio_recovers_phases():
    # Как электроспиннинг у Purushothaman 2024: β доминирует, TC ≈ 45 %
    x = np.arange(770.0, 930.0, 1.0)
    areas = {"α1": 60, "γ1": 90, "γ2": 150, "β": 900, "α2": 60, "c": 200}  # α:β:γ = 9:71:19
    y = 30 + 0.02 * x + _crystal(x, areas) + _melt_amorphous(x, area=1800)
    y = y + np.random.default_rng(5).normal(0, 0.2, x.size)
    r = fit_pvdf_melt(x, y)
    phases = 60 + 90 + 150 + 900 + 60
    assert r["tc"] == pytest.approx(100 * (phases + 200) / (phases + 200 + 1800), abs=4)
    assert r["f_beta"] == pytest.approx(100 * 900 / phases, abs=4)
    assert r["f_gamma"] == pytest.approx(100 * 240 / phases, abs=4)
    assert r["f_alpha"] + r["f_beta"] + r["f_gamma"] == pytest.approx(100)
    assert r["fit"]["r2"] > 0.999


def test_melt_own_spectrum():
    # Свой расплав: профиль с другим отношением площадей, чем в статье
    x = np.arange(770.0, 930.0, 0.5)
    own = lambda xx, area: area * (0.6 * gaussian(xx, 800, 18, 1) + 0.4 * gaussian(xx, 880, 25, 1)) \
        / (np.sqrt(np.pi / (4 * np.log(2))) * (0.6 * 18 + 0.4 * 25))
    xm = np.arange(700.0, 1000.0, 1.3)
    ym = 100 + 0.1 * xm + own(xm, 5000) + np.random.default_rng(2).normal(0, 0.1, xm.size)
    areas = {"α1": 700, "γ1": 10, "γ2": 10, "β": 200, "α2": 300, "c": 80}  # α-плёнка
    y = 10 + own(x, 1500) + _crystal(x, areas) + np.random.default_rng(4).normal(0, 0.2, x.size)
    r = fit_pvdf_melt(x, y, melt=(xm, ym))
    phases = 700 + 10 + 10 + 200 + 300
    assert r["f_alpha"] == pytest.approx(100 * 1000 / phases, abs=4)
    assert r["f_beta"] == pytest.approx(100 * 200 / phases, abs=4)
    assert r["tc"] == pytest.approx(100 * (phases + 80) / (phases + 80 + 1500), abs=4)
    assert r["table"]["phase"].iloc[-1] == "аморф. (свой расплав)"


def test_melt_template_needs_window():
    x = np.arange(780.0, 920.0)
    with pytest.raises(ValueError):
        melt_template(np.arange(800.0, 1000.0), np.ones(200), x)
    t = melt_template(np.arange(700.0, 1000.0), gaussian(np.arange(700.0, 1000.0), 850, 30, 1), x)
    assert np.trapezoid(t, x) == pytest.approx(1)
