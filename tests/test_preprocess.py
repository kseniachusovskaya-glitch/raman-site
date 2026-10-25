"""Тесты для ramancore/preprocess.py на синтетических спектрах."""

import numpy as np
import pytest

from ramancore.preprocess import crop, remove_spikes


def lorentz(x, center, fwhm, amplitude):
    """Лоренцев пик: так выглядит, например, пик c-Si."""
    return amplitude / (1 + ((x - center) / (fwhm / 2)) ** 2)


def noisy_si(step=0.5, seed=0):
    """Синтетический спектр: пик Si 520.7 см⁻¹ + фон + шум."""
    x = np.arange(100, 1000, step)
    rng = np.random.default_rng(seed)
    y = lorentz(x, 520.7, 4, 1000) + 50 + rng.normal(0, 5, x.size)
    return x, y


# ---------- crop ----------

def test_crop_keeps_range_inclusive():
    x = np.array([50, 100, 200, 3200, 3300])
    y = np.array([1, 2, 3, 4, 5])
    xc, yc = crop(x, y)  # по умолчанию 100–3200
    assert list(xc) == [100, 200, 3200]
    assert list(yc) == [2, 3, 4]


def test_crop_custom_range():
    x, y = noisy_si()
    xc, yc = crop(x, y, 400, 600)
    assert xc.min() >= 400 and xc.max() <= 600
    assert len(xc) == len(yc)


def test_crop_empty_range_raises():
    x, y = noisy_si()
    with pytest.raises(ValueError):
        crop(x, y, 5000, 6000)


# ---------- remove_spikes ----------

def test_removes_single_point_spike():
    x, y = noisy_si()
    clean = y.copy()
    i = np.searchsorted(x, 800)  # индекс точки около 800 см⁻¹
    y[i] += 3000  # космический луч в 1 точку
    result = remove_spikes(y)
    assert abs(result[i] - clean[i]) < 30  # луч убран, осталось значение на уровне шума


def test_removes_two_point_spike():
    x, y = noisy_si()
    clean = y.copy()
    i = np.searchsorted(x, 300)
    y[i] += 2000
    y[i + 1] += 2500
    result = remove_spikes(y)
    assert np.all(np.abs(result[i : i + 2] - clean[i : i + 2]) < 30)


@pytest.mark.parametrize("step", [0.5, 1.0])
def test_keeps_strong_narrow_si_peak(step):
    # Главное требование: настоящий пик Si не трогаем
    x, y = noisy_si(step=step)
    result = remove_spikes(y)
    assert np.array_equal(result, y)


def test_no_false_spikes_in_pure_noise():
    rng = np.random.default_rng(1)
    y = 100 + rng.normal(0, 5, 3000)
    assert np.array_equal(remove_spikes(y), y)


def test_input_not_modified():
    x, y = noisy_si()
    y[100] += 3000
    before = y.copy()
    remove_spikes(y)
    assert np.array_equal(y, before)
