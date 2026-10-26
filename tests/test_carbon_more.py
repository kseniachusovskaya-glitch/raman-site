"""Тесты: число слоёв графена, дефекты, метрики Backes, 5 полос сажи, La."""

import numpy as np
import pytest

from ramancore.carbon import (
    GRAPHENE_BILAYER_SPLIT, backes_flakes, backes_m1, backes_m2, crystallite_size_la, fit_soot,
    graphene_defects, graphene_layers,
)
from ramancore.peaks import gaussian, lorentzian

X = np.arange(2450, 2950, 1.0)
RNG = np.random.default_rng(1)


def two_d(centers, widths, heights, noise=3):
    y = sum(lorentzian(X, c, w, h) for c, w, h in zip(centers, widths, heights)) + 50
    return y + RNG.normal(0, noise, X.size)


def test_monolayer():
    y = two_d([2680], [28], [4000])
    r = graphene_layers(X, y, 2680, height_g=1000, laser_nm=514.5)
    assert r["verdict"].startswith("Монослой")
    assert r["fwhm_1"] == pytest.approx(28, abs=1)
    assert r["i2d_ig"] == pytest.approx(4, rel=0.05)


def test_bilayer_is_asymmetric():
    c = 2690 + np.array(GRAPHENE_BILAYER_SPLIT[514.5])
    y = two_d(c, [24] * 4, [300, 1000, 1000, 300])
    r = graphene_layers(X, y, 2690, height_g=1000, laser_nm=514.5)
    assert r["rss_ratio"] < 0.5
    assert "2–5 слоёв" in r["verdict"]


def test_graphite_two_components():
    y = two_d([2685, 2725], [40, 40], [250, 500])
    r = graphene_layers(X, y, 2715, height_g=1000, laser_nm=514.5)
    assert "больше ~5 слоёв" in r["verdict"]


def test_turbostratic_is_wide_symmetric():
    y = two_d([2700], [50], [1500])
    r = graphene_layers(X, y, 2700, height_g=1000, laser_nm=514.5)
    assert "турбостратный" in r["verdict"]


def test_defects_formula():
    # I_D/I_G = 1 при 514 нм: L_D² = 1.8e-9·514⁴ ≈ 125 нм² → L_D ≈ 11.2 нм
    r = graphene_defects(1.0, 514)
    assert r["ld"] == pytest.approx(np.sqrt(1.8e-9 * 514**4), rel=1e-9)
    assert r["nd"] == pytest.approx(1.8e22 / 514**4, rel=1e-9)
    # L_D² ·n_D = 1.8e-9·1.8e22 нм²/см² — не зависит от лазера
    assert r["ld"] ** 2 * r["nd"] == pytest.approx(1.8e-9 * 1.8e22)
    assert not any("< 10 нм" in w for w in r["warnings"])


def test_defects_warnings():
    assert any("< 10 нм" in w for w in graphene_defects(4.0, 514)["warnings"])
    assert any("стадию 2" in w for w in graphene_defects(1.0, 514, fwhm_g=60)["warnings"])
    assert np.isnan(graphene_defects(0.0, 514)["ld"])


def test_backes_formulas():
    r = backes_flakes(fwhm_g=20, id_ig=0.4, id_ig_graphite=0.1, i2d_ig=0.6)
    assert r["l_width"][0] == pytest.approx(50 * np.exp(-4.2))
    assert r["l_d"][0] == pytest.approx(0.094 / 0.3)
    assert r["n_m1"][0] == pytest.approx(1.04 * 0.6 ** -2.32)
    # Монослой в этой метрике: M1 ≈ 1 → N ≈ 1 (так в статье)
    assert backes_flakes(i2d_ig=1.0)["n_m1"][0] == pytest.approx(1.04)


def test_backes_liquid_keeps_only_width():
    r = backes_flakes(fwhm_g=20, id_ig=0.4, id_ig_graphite=0.1, i2d_ig=0.6, liquid=True)
    assert "l_width" in r and "l_d" not in r and "n_m1" not in r


def test_backes_m2_same_spectrum_is_one():
    y = two_d([2685, 2725], [40, 40], [250, 500], noise=0)
    m2, w1 = backes_m2(X, y, X, y)
    assert m2 == pytest.approx(1.0)
    assert w1 == pytest.approx(2723, abs=3)
    # У тонкого графена 2D симметрична и ниже по частоте:
    # I(ω1)/I(ω1−30) меньше, чем у графита → M2 < 1
    thin = two_d([2700], [40], [800], noise=0)
    assert backes_m2(X, thin, X, y)[0] < 1


def soot(noise=5):
    x = np.arange(700, 2100, 1.5)
    y = (lorentzian(x, 1590, 70, 800) + lorentzian(x, 1345, 180, 1500)
         + lorentzian(x, 1615, 40, 250) + gaussian(x, 1510, 160, 300)
         + lorentzian(x, 1190, 200, 150) + 100 + 0.05 * (x - 1400))
    return x, y + np.random.default_rng(2).normal(0, noise, x.size)


def test_soot_five_bands():
    r = fit_soot(*soot())
    b = r["bands"].set_index("band")
    assert b.loc["G", "center"] == pytest.approx(1590, abs=5)
    assert b.loc["D1", "center"] == pytest.approx(1345, abs=5)
    assert b.loc["D1", "fwhm"] == pytest.approx(180, rel=0.1)
    assert b.loc["D3", "center"] == pytest.approx(1510, abs=20)
    # Отношение площадей D1/G: (π/2·180·1500)/(π/2·70·800)
    assert b.loc["D1", "area_to_g"] == pytest.approx(180 * 1500 / (70 * 800), rel=0.1)


def test_la_formula_and_range():
    la, w = crystallite_size_la(1.0, 514.5)
    assert la == pytest.approx(2.4e-10 * 514.5**4)  # ≈ 16.8 нм
    assert any("меньше 20 нм" in s for s in w)
    la, w = crystallite_size_la(0.2, 514.5)
    assert la == pytest.approx(84, rel=0.01) and not any("вне калибровки" in s for s in w)
    assert any("457.9–647" in s for s in crystallite_size_la(0.2, 785)[1])


def test_backes_m1_uses_maxima():
    x = np.arange(1400, 2950, 1.0)
    y = lorentzian(x, 1582, 20, 1000) + lorentzian(x, 2690, 60, 400) + lorentzian(x, 2720, 40, 400) + 100
    expected = (lorentzian(x, 2690, 60, 400) + lorentzian(x, 2720, 40, 400)).max() / 1000
    assert backes_m1(x, y) == pytest.approx(expected, rel=0.03)
