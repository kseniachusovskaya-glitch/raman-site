"""Тесты для ramancore/thermal.py на синтетических данных."""

import numpy as np
import pytest

from ramancore.peaks import fit_peaks
from ramancore.thermal import balkanski, fit_temperature_shift, temperature_from_shift

from conftest import make_spectrum

# Синтетический «материал»: параметры взяты произвольно, близко к LO GaAs
W0, A = 296.0, -1.5
TEMPS = np.array([227, 254, 274, 294, 334, 374])


def test_model_decreases_with_temperature():
    w = balkanski(TEMPS, W0, A)
    assert np.all(np.diff(w) < 0)
    assert balkanski(1.0, W0, A) == pytest.approx(W0 + A)  # при T → 0 остаётся ω0 + A


def test_calibration_recovers_parameters():
    cal = fit_temperature_shift(TEMPS, balkanski(TEMPS, W0, A))
    assert cal["omega0"] == pytest.approx(W0, abs=1e-3)
    assert cal["A"] == pytest.approx(A, abs=1e-3)
    assert cal["slope"] < 0 and cal["T_range"] == (227, 374)


def test_temperature_roundtrip_and_error():
    cal = fit_temperature_shift(TEMPS, balkanski(TEMPS, W0, A))
    T, T_err, warnings = temperature_from_shift(balkanski(310, W0, A), cal, omega_err=0.1)
    assert T == pytest.approx(310, abs=0.01)
    assert T_err == pytest.approx(0.1 / abs(cal["slope"]), rel=0.3)
    assert not any("экстраполяция" in w for w in warnings)


def test_extrapolation_warning():
    cal = fit_temperature_shift(TEMPS, balkanski(TEMPS, W0, A))
    T, _, warnings = temperature_from_shift(balkanski(600, W0, A), cal)
    assert T == pytest.approx(600, abs=0.1)
    assert any("экстраполяция" in w for w in warnings)


def test_needs_three_points():
    with pytest.raises(ValueError):
        fit_temperature_shift([300, 350], [295, 294])


@pytest.mark.parametrize("step", [1.21, 6.24])
def test_full_chain_on_synthetic_spectra(step):
    # Серия спектров → центры пиков → калибровка → температура «неизвестного»
    centers, errs = [], []
    for i, T in enumerate(TEMPS):
        x, y = make_spectrum([(balkanski(T, W0, A), 9, 1000)], step=step, seed=i, xmax=600)
        table, _ = fit_peaks(x, y, [balkanski(T, W0, A)], shape="lorentz")
        centers.append(table.center[0])
        errs.append(table.center_err[0])
    cal = fit_temperature_shift(TEMPS, centers, errs)
    x, y = make_spectrum([(balkanski(320, W0, A), 9, 1000)], step=step, seed=99, xmax=600)
    table, _ = fit_peaks(x, y, [balkanski(320, W0, A)], shape="lorentz")
    T, T_err, _ = temperature_from_shift(table.center[0], cal, table.center_err[0])
    assert T == pytest.approx(320, abs=max(15, 3 * T_err))


def test_article_parameters_give_published_slope():
    # Максимов и др., Квантовая электроника 56(4), 247 (2026): ω0 = 302.0,
    # A = −0.93, B = −0.11. Без члена теплового расширения наклон 273–373 K
    # около −0.018 см⁻¹/К (в статье с расширением −0.022)
    slope = (balkanski(373, 302.0, -0.93, -0.11) - balkanski(273, 302.0, -0.93, -0.11)) / 100
    assert slope == pytest.approx(-0.018, abs=0.001)


def test_four_phonon_fit_recovers_parameters():
    T = np.linspace(80, 700, 15)  # для члена B нужен широкий диапазон
    cal = fit_temperature_shift(T, balkanski(T, 302.0, -0.93, -0.11), four_phonon=True)
    assert cal["A"] == pytest.approx(-0.93, abs=1e-3)
    assert cal["B"] == pytest.approx(-0.11, abs=1e-3)
    T_back, _, _ = temperature_from_shift(balkanski(500, 302.0, -0.93, -0.11), cal)
    assert T_back == pytest.approx(500, abs=0.1)
