"""Тесты для расчёта напряжений (si_stress, volume_change)."""

import numpy as np
import pytest

from ramancore.peaks import fit_peaks
from ramancore.semiconductors import si_stress, volume_change

from conftest import make_spectrum


def test_si_sign_and_value():
    # +1 см⁻¹ — сжатие −435 МПа; −0.5 см⁻¹ — растяжение +217.5 МПа
    assert si_stress(521.7, 520.7)[0] == pytest.approx(-435)
    assert si_stress(520.2, 520.7)[0] == pytest.approx(217.5)
    assert si_stress(520.7, 520.7, omega_err=0.05)[1] == pytest.approx(21.75)


def test_si_stress_from_synthetic_spectra():
    # Эталон и «напряжённый» образец (−300 МПа → пик выше на 300/435 см⁻¹)
    shift = 300 / 435
    x, y = make_spectrum([(520.7, 4, 1000)], step=0.56, xmin=400, xmax=650)
    ref, _ = fit_peaks(x, y, [520.7])
    x, y = make_spectrum([(520.7 + shift, 4, 1000)], step=0.56, seed=1, xmin=400, xmax=650)
    meas, _ = fit_peaks(x, y, [521.4])
    sigma, err = si_stress(meas.center[0], ref.center[0], meas.center_err[0])
    assert sigma == pytest.approx(-300, abs=max(15, 3 * err))


def test_volume_change_gaas():
    # Δω/ω0 = −γ·ΔV/V: при γ(LO) = 1.23 сжатие на 1 % поднимает LO на 1.23 %
    w0 = 291.2
    dV, _ = volume_change(w0 * 1.0123, w0, "GaAs", "LO")
    assert dV == pytest.approx(-1.0, abs=1e-6)
    assert np.all(volume_change([w0 - 1, w0 + 1], w0)[0] * [1, -1] > 0)


def test_volume_change_unknown_material():
    with pytest.raises(ValueError):
        volume_change(200, 210, "InSb")
