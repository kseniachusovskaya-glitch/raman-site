"""Тесты для ramancore/preprocess.py на синтетических спектрах."""

import numpy as np
import pytest

from ramancore.preprocess import crop, remove_spikes, smooth, subtract_baseline


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


# ---------- smooth ----------

def test_smooth_reduces_noise():
    rng = np.random.default_rng(2)
    y = 100 + rng.normal(0, 5, 2000)
    # окно 5 убирает ~30% шума, окно 11 — ~45%
    assert np.std(smooth(y)) < 0.8 * np.std(y)
    assert np.std(smooth(y, window=11)) < 0.6 * np.std(y)


def test_smooth_keeps_si_peak_position():
    # Требование этапа 1: центр пика после обработки — с точностью 0.5 см⁻¹
    x, y = noisy_si(step=0.5)
    ys = smooth(y)
    assert x[np.argmax(ys)] == pytest.approx(520.7, abs=0.5)


def test_smooth_keeps_height_when_window_narrower_than_peak():
    x = np.arange(100, 1000, 0.5)
    y = lorentz(x, 520.7, 4, 1000)  # FWHM 4 см⁻¹ = 8 точек при шаге 0.5
    assert smooth(y).max() > 0.9 * y.max()


def test_smooth_preserves_straight_line():
    # Полином 3-й степени точно воспроизводит прямую
    y = np.linspace(0, 100, 50)
    assert np.allclose(smooth(y), y)


def test_smooth_short_spectrum_and_even_window():
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    assert len(smooth(y, window=11)) == len(y)  # 7 точек < окна 11
    assert len(smooth(y)) == len(y)
    assert len(smooth(np.arange(20.0), window=10)) == 20  # чётное окно → 9
    short = np.array([1.0, 5.0, 1.0])
    assert np.array_equal(smooth(short), short)  # слишком коротко — без изменений


# ---------- subtract_baseline ----------

def curved_background_spectrum(step, seed=0):
    """Пик Si на изогнутом фоне (как люминесценция) + шум σ = 5."""
    x = np.arange(100, 1000, step)
    background = 1e-4 * (x - 500) ** 2 + 50
    y = background + lorentz(x, 520.7, 6, 1000) + np.random.default_rng(seed).normal(0, 5, x.size)
    return x, y, background


@pytest.mark.parametrize("step", [0.5, 6.24])
@pytest.mark.parametrize("method", ["arpls", "asls", "poly"])
def test_baseline_follows_curved_background(step, method):
    x, y, background = curved_background_spectrum(step)
    # asls жёстче arpls: на коротком спектре (~150 точек) ему нужно меньшее lam
    lam = 1e3 if (method == "asls" and step > 1) else 1e5
    corrected, baseline = subtract_baseline(x, y, method=method, lam=lam)
    far = np.abs(x - 520.7) > 40  # точки вдали от пика
    # arpls учитывает шум и идёт посередине; asls и poly прижимаются
    # к нижней границе шума, поэтому у них допуск больше (~4σ)
    tolerance = 15 if method == "arpls" else 20
    assert np.max(np.abs(baseline - background)[far]) < tolerance


@pytest.mark.parametrize("step", [0.5, 6.24])
def test_baseline_keeps_peak_height(step):
    x, y, background = curved_background_spectrum(step)
    corrected, _ = subtract_baseline(x, y)
    i = np.argmax(corrected)
    true_height = lorentz(x[i], 520.7, 6, 1000)
    assert corrected[i] == pytest.approx(true_height, rel=0.05)


def test_baseline_shapes_and_input_not_modified():
    x, y, _ = curved_background_spectrum(1.0)
    before = y.copy()
    corrected, baseline = subtract_baseline(x, y)
    assert corrected.shape == baseline.shape == y.shape
    assert np.allclose(corrected + baseline, y)
    assert np.array_equal(y, before)


def test_baseline_unknown_method_raises():
    x, y, _ = curved_background_spectrum(1.0)
    with pytest.raises(ValueError):
        subtract_baseline(x, y, method="magic")
