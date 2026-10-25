"""Тесты для ramancore/peaks.py на синтетических спектрах с известными пиками."""

import numpy as np
import pytest

from ramancore.peaks import (
    curves_on_grid, find_peaks, fit_peaks, gaussian, lorentzian, noise_level, peak_area, pseudo_voigt,
)


def spectrum(peaks, step=0.5, noise=5.0, slope=0.05, shape=lorentzian, seed=0):
    """Синтетический спектр: сумма пиков + наклонный фон + шум.

    peaks — список (center, fwhm, amplitude).
    """
    x = np.arange(100, 1000, step)
    y = 50 + slope * x
    for c, w, a in peaks:
        y = y + shape(x, c, w, a)
    y = y + np.random.default_rng(seed).normal(0, noise, x.size)
    return x, y


# ---------- формы и площади ----------

@pytest.mark.parametrize("shape, func", [("lorentz", lorentzian), ("gauss", gaussian)])
def test_area_formula_matches_numeric_integral(shape, func):
    x = np.linspace(-5000, 5000, 2_000_001)
    numeric = np.trapezoid(func(x, 0, 8, 100), x)
    assert peak_area(8, 100, shape) == pytest.approx(numeric, rel=1e-3)


def test_profiles_have_given_fwhm():
    for f in (lorentzian, gaussian):
        assert f(4.0, 0, 8, 1) == pytest.approx(0.5)  # на полуширине — половина высоты
    assert pseudo_voigt(4.0, 0, 8, 1, 0.3) == pytest.approx(0.5)


# ---------- find_peaks ----------

def test_find_peaks_finds_known_peaks():
    x, y = spectrum([(268, 8, 1000), (292, 8, 400), (520.7, 4, 800)])
    found = find_peaks(x, y)
    for c in (268, 292, 520.7):
        assert np.min(np.abs(found - c)) <= 1.0
    assert len(found) == 3  # и ничего лишнего в шуме


def test_noise_level_ignores_peaks():
    x, y = spectrum([(520.7, 4, 1000)], noise=5)
    assert noise_level(y) == pytest.approx(5, rel=0.15)


@pytest.mark.parametrize("seed", range(5))
def test_find_peaks_no_false_peaks_in_noise(seed):
    x, y = spectrum([(520.7, 4, 1000)], noise=20, seed=seed)
    assert len(find_peaks(x, y)) == 1


# ---------- fit_peaks ----------

def test_fit_single_lorentz_fine_step():
    x, y = spectrum([(520.7, 4, 1000)])
    table, fit = fit_peaks(x, y, [521])
    r = table.iloc[0]
    assert r.center == pytest.approx(520.7, abs=0.05)
    assert r.fwhm == pytest.approx(4, rel=0.03)
    assert r.area == pytest.approx(peak_area(4, 1000), rel=0.03)
    assert fit["r2"] > 0.99


def test_fit_coarse_step_like_real_gaas():
    # Шаг 6.24 см⁻¹, как в спектре GaAs: на пик всего 2–3 точки,
    # но центр всё равно должен находиться с точностью 0.5 см⁻¹
    x, y = spectrum([(268.3, 10, 90000)], step=6.24, noise=500, slope=5)
    found = find_peaks(x, y)
    table, _ = fit_peaks(x, y, found)
    r = table.iloc[0]
    assert r.center == pytest.approx(268.3, abs=0.5)
    assert r.fwhm == pytest.approx(10, rel=0.1)
    assert r.area == pytest.approx(peak_area(10, 90000), rel=0.1)


def test_fit_two_overlapping_peaks_with_sloped_background():
    # TO и LO GaAs (ориентировочно 268 и 292 см⁻¹) подгоняются вместе
    x, y = spectrum([(268, 8, 1000), (292, 8, 400)], step=1.0, slope=0.5)
    table, fit = fit_peaks(x, y, [268, 292])
    assert list(table.center.round(0)) == [268, 292]
    assert table.amplitude.iloc[0] == pytest.approx(1000, rel=0.05)
    assert table.amplitude.iloc[1] == pytest.approx(400, rel=0.05)
    assert len(fit["components"]) == 2


def test_fit_gauss_and_voigt():
    x, y = spectrum([(500, 10, 1000)], shape=gaussian)
    tg, _ = fit_peaks(x, y, [500], shape="gauss")
    assert tg.fwhm.iloc[0] == pytest.approx(10, rel=0.03)

    x, y = spectrum([(500, 10, 1000)], shape=lambda x, c, w, a: pseudo_voigt(x, c, w, a, 0.4))
    tv, _ = fit_peaks(x, y, [500], shape="voigt")
    assert tv.eta.iloc[0] == pytest.approx(0.4, abs=0.1)
    assert tv.fwhm.iloc[0] == pytest.approx(10, rel=0.05)


def test_errors_are_positive_and_reasonable():
    x, y = spectrum([(520.7, 4, 1000)], noise=20)
    table, _ = fit_peaks(x, y, [521])
    r = table.iloc[0]
    for col in ("center_err", "fwhm_err", "amplitude_err", "area_err"):
        assert 0 < r[col] < np.inf
    # Истинный центр должен быть в пределах ~3σ от найденного
    assert abs(r.center - 520.7) < 3 * r.center_err + 0.01


def test_too_few_points_raises():
    x, y = spectrum([(500, 10, 1000)], step=6.24)
    with pytest.raises(ValueError):
        fit_peaks(x, y, [500], margin=5)


def test_unknown_shape_raises():
    x, y = spectrum([(500, 10, 1000)])
    with pytest.raises(ValueError):
        fit_peaks(x, y, [500], shape="triangle")


def test_curves_on_grid_match_fit_on_data_points():
    x, y = spectrum([(268, 8, 1000), (292, 8, 400)], step=1.0)
    for shape in ("lorentz", "voigt"):
        table, fit = fit_peaks(x, y, [268, 292], shape=shape)
        baseline, comps, total = curves_on_grid(table, fit, fit["x"])
        assert np.allclose(total, fit["model"])
        assert np.allclose(baseline, fit["baseline"])
        assert len(comps) == 2
