"""Тесты для ramancore/semiconductors.py на синтетических спектрах."""

import numpy as np
import pytest

from ramancore.semiconductors import gaas_orientation

from conftest import make_spectrum


def gaas(to_height, lo_height, step=1.0, seed=0, to=268.0, lo=291.5):
    # Ширина линии не меньше ~1.5 шага: прибор с шагом 6.24 см⁻¹ не даёт
    # линий уже ~10 см⁻¹ (аппаратное уширение)
    fwhm = max(4.0, 1.5 * step)
    return make_spectrum([(to, fwhm, to_height), (lo, fwhm, lo_height)],
                         step=step, noise=5, seed=seed, xmax=1000)


@pytest.mark.parametrize("step", [1.0, 6.24])
def test_100_lo_only(step):
    # (100): TO запрещена — оставим лишь слабую «утечку» 5 %
    x, y = gaas(50, 1000, step=step)
    r = gaas_orientation(x, y)
    assert r["orientation"] == "(100)"
    assert r["ratio"] == pytest.approx(0.05, abs=0.03)


@pytest.mark.parametrize("step", [1.0, 6.24])
def test_110_to_only(step):
    x, y = gaas(1000, 50, step=step)
    assert gaas_orientation(x, y)["orientation"] == "(110)"


def test_111_both():
    x, y = gaas(1000, 800)
    r = gaas_orientation(x, y)
    assert r["orientation"] == "(111) / смешанная"
    assert r["ratio"] == pytest.approx(1.25, rel=0.1)
    assert 0 < r["ratio_err"] < 0.2


def test_warning_when_peaks_shifted():
    # Сильно сдвинутые пики (например, напряжения или не GaAs)
    x, y = gaas(1000, 800, to=262, lo=285)
    r = gaas_orientation(x, y)
    assert "далеко от" in " ".join(r["warnings"])
    assert list(r["table"].center.round(0)) == [262, 285]  # нашёл сдвинутые пики


def test_doping_warning_always_present():
    x, y = gaas(50, 1000)
    assert any("плазмон" in w for w in gaas_orientation(x, y)["warnings"])


def test_weak_unresolved_peak_gives_warning():
    # Слишком узкие для шага 6.24 линии: подгонка не должна выдавать уверенный ответ молча
    x, y = make_spectrum([(268, 4, 50), (291.5, 4, 1000)], step=6.24, noise=5, xmax=1000)
    r = gaas_orientation(x, y)
    if r["orientation"] == "не определена":
        assert "не сошлась" in r["warnings"][0]
    elif r["ratio_err"] > r["ratio"]:
        assert any("не разрешены" in w for w in r["warnings"])


def test_not_gaas():
    # Спектр Si: в районе 268–292 см⁻¹ только шум
    x, y = make_spectrum([(520.7, 4, 1000)], noise=5, xmax=1000)
    r = gaas_orientation(x, y)
    assert r["orientation"] == "не определена"
    assert "не GaAs" in r["warnings"][0]
