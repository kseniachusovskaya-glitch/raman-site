"""Тесты для ramancore/semiconductors.py на синтетических спектрах."""

import numpy as np
import pytest

from ramancore.semiconductors import (
    LOPC_PARAMS, concentration_from_mode, coupled_modes, gaas_orientation,
    lplus_concentration, plasma_frequency,
)

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


# ---------- концентрация по связанной моде L+ ----------

def test_plasma_frequency_scales_as_sqrt_n():
    p = LOPC_PARAMS["GaAs"]
    w1 = plasma_frequency(1e17, p["eps_inf"], p["m_eff"])
    w4 = plasma_frequency(4e17, p["eps_inf"], p["m_eff"])
    assert w4 == pytest.approx(2 * w1)


def test_coupled_modes_limits():
    p = LOPC_PARAMS["GaAs"]
    lm, lp, _ = coupled_modes(1e12)            # почти нет электронов
    assert lm == pytest.approx(0, abs=1) and lp == pytest.approx(p["LO"], abs=0.1)
    lm, lp, wp = coupled_modes(1e20)           # очень много электронов
    assert lm == pytest.approx(p["TO"], abs=0.5) and lp == pytest.approx(wp, rel=0.01)


@pytest.mark.parametrize("material", ["GaAs", "InAs", "InSb"])
@pytest.mark.parametrize("n", [3e17, 1.5e18])
def test_inverse_gives_back_concentration(material, n):
    lm, lp, _ = coupled_modes(n, material)
    assert concentration_from_mode(lp, material)[0] == pytest.approx(n, rel=1e-6)
    assert concentration_from_mode(lm, material)[0] == pytest.approx(n, rel=1e-6)


def test_frequency_between_to_and_lo_rejected():
    with pytest.raises(ValueError):
        concentration_from_mode(280.0, "GaAs")


def test_overrides_use_measured_phonons():
    # Свои TO/LO, измеренные на нелегированном образце, меняют ответ
    _, lp, _ = coupled_modes(1e18, TO=270.0, LO=294.0)
    assert concentration_from_mode(lp, TO=270.0, LO=294.0)[0] == pytest.approx(1e18, rel=1e-6)


@pytest.mark.parametrize("step", [1.21, 6.24])
def test_lplus_found_in_synthetic_n_gaas(step):
    n = 1.5e18
    lm, lp, _ = coupled_modes(n)
    # TO + L− рядом с TO, слабая LO из обеднённого слоя и широкая слабая L+
    x, y = make_spectrum([(lm, 6, 800), (291.2, 5, 300)], step=step, noise=5, xmax=1000)
    y = y + 60 * np.exp(-4 * np.log(2) * ((x - lp) / 100) ** 2)
    r = lplus_concentration(x, y)
    assert r["omega"] == pytest.approx(lp, abs=5)
    assert r["n"] == pytest.approx(n, rel=0.05)


def test_no_lplus_in_undoped():
    x, y = make_spectrum([(291.2, 5, 1000)], noise=5, xmax=1000)
    r = lplus_concentration(x, y)
    assert np.isnan(r["n"]) and "не видно" in r["warnings"][0]
