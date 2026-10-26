"""Тесты для intrinsic_fwhm (peaks.py) и phonon_quality (semiconductors.py)."""

import numpy as np
import pytest
from scipy.special import voigt_profile

from ramancore.peaks import intrinsic_fwhm
from ramancore.semiconductors import phonon_quality

from conftest import make_spectrum


def _fwhm_numeric(x, y):
    """Ширина на полувысоте по точкам (для проверки формулы)."""
    above = x[y >= y.max() / 2]
    return above[-1] - above[0]


def test_no_instrument_broadening_returns_measured():
    assert intrinsic_fwhm(8.0, 0.0) == pytest.approx(8.0, rel=1e-3)


def test_narrower_than_instrument_gives_zero():
    assert intrinsic_fwhm(6.0, 7.0) == 0.0
    assert np.array_equal(intrinsic_fwhm([5.0, 7.0], 7.0), [0.0, 0.0])


@pytest.mark.parametrize("fL", [3.0, 7.0, 15.0])
def test_recovers_lorentz_width_of_numeric_voigt(fL):
    # Свёртка лоренциана fL с гауссианом fG = 7 см⁻¹ (как у Confotec Duo)
    fG = 7.0
    x = np.arange(-200, 200, 0.001)
    sigma, gamma = fG / (2 * np.sqrt(2 * np.log(2))), fL / 2
    fV = _fwhm_numeric(x, voigt_profile(x, sigma, gamma))
    assert intrinsic_fwhm(fV, fG) == pytest.approx(fL, rel=0.02)


def _gaas_100(to_height, lo_height, lo_center=292.0, seed=0):
    # TO и LO с одинаковой шириной: доля площадей = доля высот
    return make_spectrum([(268.0, 8, to_height), (lo_center, 8, lo_height)],
                         step=1.21, noise=3.0, seed=seed, xmin=150, xmax=450)


def test_lo_fraction_of_good_crystal():
    x, y = _gaas_100(to_height=100, lo_height=900)
    res = phonon_quality(x, y, "GaAs")
    assert res["lo_fraction"] == pytest.approx(0.9, abs=0.03)
    assert res["lo_fraction_err"] < 0.03
    assert list(res["table"].index) == ["TO", "LO"]


def test_lo_fraction_of_disordered_crystal():
    x, y = _gaas_100(to_height=800, lo_height=200)
    assert phonon_quality(x, y, "GaAs")["lo_fraction"] == pytest.approx(0.2, abs=0.03)


def test_lo_shift_from_reference():
    x, y = _gaas_100(to_height=300, lo_height=900, lo_center=289.0)
    res = phonon_quality(x, y, "GaAs", reference_lo=292.0)
    assert res["lo_shift"] == pytest.approx(-3.0, abs=0.3)
    assert np.isnan(phonon_quality(x, y, "GaAs")["lo_shift"])


def test_intrinsic_column_with_instrument_width():
    x, y = _gaas_100(to_height=300, lo_height=900)
    res = phonon_quality(x, y, "GaAs", instrument_fwhm=7.0)
    t = res["table"]
    assert "intrinsic_fwhm" in t.columns
    assert np.all(t.intrinsic_fwhm < t.fwhm)
    assert "intrinsic_fwhm" not in phonon_quality(x, y, "GaAs")["table"].columns
