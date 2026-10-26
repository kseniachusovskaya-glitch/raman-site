"""Тесты для нанотрубок: диаметр по RBM и по расщеплению G."""

import numpy as np
import pytest

from ramancore.carbon import bwf, find_rbm, fit_cnt_g, rbm_diameter
from ramancore.peaks import lorentzian


def test_rbm_models():
    assert rbm_diameter(227.0, "pristine") == pytest.approx(1.0)
    assert rbm_diameter(248.0, "jorio") == pytest.approx(1.0)
    # Обратная проверка формулы Araujo: ω(dt) = 227/dt·√(1 + C·dt²)
    for dt in (0.8, 1.2, 2.0):
        w = 227 / dt * np.sqrt(1 + 0.05786 * dt**2)
        assert rbm_diameter(w, "environment") == pytest.approx(dt, rel=1e-9)
    # Окружение повышает частоту: при той же ω диаметр получается больше
    assert rbm_diameter(200, "environment") > rbm_diameter(200, "pristine")


def test_find_rbm():
    x = np.arange(80, 450, 0.8)
    y = lorentzian(x, 180, 6, 500) + lorentzian(x, 260, 6, 300) + 50 + 0.05 * x
    y = y + np.random.default_rng(0).normal(0, 3, x.size)
    t = find_rbm(x, y)
    assert len(t) == 2
    assert t.omega.iloc[0] == pytest.approx(180, abs=1.5)
    assert t.dt_pristine.iloc[0] == pytest.approx(227 / 180, rel=0.01)


def g_band(minus_shape="lorentz", split=20, seed=0):
    x = np.arange(1400, 1720, 0.8)
    y = lorentzian(x, 1591, 12, 1000) + 100
    if minus_shape == "lorentz":
        y = y + lorentzian(x, 1591 - split, 15, 350)
    else:
        y = y + bwf(x, 1591 - split, 60, 350, -3)
    return x, y + np.random.default_rng(seed).normal(0, 5, x.size)


def test_semiconducting_g():
    r = fit_cnt_g(*g_band("lorentz", split=20))
    assert r["kind"] == "semiconducting"
    assert r["split"] == pytest.approx(20, abs=1)
    assert r["dt"] == pytest.approx(np.sqrt(47.7 / 20), rel=0.05)


def test_metallic_g_is_bwf():
    r = fit_cnt_g(*g_band("bwf", split=40))
    assert r["kind"] == "metallic"
    assert r["q"] == pytest.approx(-3, rel=0.3)
