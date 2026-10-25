"""Поиск пиков и их аппроксимация: центр, FWHM, площадь — с погрешностями.

Подгонка идёт по ИСХОДНЫМ (несглаженным) точкам: сглаживание искажает
высоту и ширину узких пиков. Фон под пиками подгоняется вместе с ними
прямой линией, поэтому заранее вычитать базовую линию не обязательно.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.signal import peak_widths

# Коэффициент для площади гауссиана: площадь = A · FWHM · sqrt(π / (4 ln 2))
_GAUSS_AREA = np.sqrt(np.pi / (4 * np.log(2)))


# ---------- формы пиков ----------

def lorentzian(x, center, fwhm, amplitude):
    """Лоренциан: естественная форма фононной линии."""
    return amplitude / (1 + ((x - center) / (fwhm / 2)) ** 2)


def gaussian(x, center, fwhm, amplitude):
    """Гауссиан: уширение из-за неоднородности образца или прибора."""
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2)


def pseudo_voigt(x, center, fwhm, amplitude, eta):
    """Псевдо-Фойгт: смесь eta·Лоренц + (1 − eta)·Гаусс с общими центром и FWHM.

    Приближение к профилю Фойгта (свёртка лоренцевой линии с гауссовой
    аппаратной функцией). eta = 1 — чистый Лоренц, eta = 0 — чистый Гаусс.
    """
    return eta * lorentzian(x, center, fwhm, amplitude) + (1 - eta) * gaussian(x, center, fwhm, amplitude)


def peak_area(fwhm, amplitude, shape="lorentz", eta=1.0):
    """Аналитическая площадь пика по его ширине и высоте."""
    lorentz_area = np.pi / 2 * amplitude * fwhm
    gauss_area = _GAUSS_AREA * amplitude * fwhm
    if shape == "lorentz":
        return lorentz_area
    if shape == "gauss":
        return gauss_area
    return eta * lorentz_area + (1 - eta) * gauss_area


# Сколько параметров у одного пика каждой формы
_N_PARAMS = {"lorentz": 3, "gauss": 3, "voigt": 4}
_PROFILES = {"lorentz": lorentzian, "gauss": gaussian, "voigt": pseudo_voigt}


# ---------- поиск пиков ----------

def noise_level(y):
    """Оценка шума σ по разностям соседних точек (устойчиво к пикам).

    Для белого шума разность двух точек имеет σ·√2; MAD/0.6745 — устойчивая
    оценка σ, на которую почти не влияют пики.
    """
    d = np.diff(np.asarray(y, dtype=float))
    return np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2)


def find_peaks(x, y, prominence=0.02, min_snr=8):
    """Находит пики и возвращает их положения x (по возрастанию).

    Пик засчитывается, если его «выступаемость» над соседним фоном больше
    ОБОИХ порогов:
      prominence — доля от размаха сигнала (max − min), 0.02 = 2 %;
      min_snr    — во сколько раз больше уровня шума (см. noise_level).
                   8 выбрано так: у чистого шума из ~2000 точек самый
                   «выступающий» всплеск доходит до ~7σ.
    Второй порог не даёт принимать всплески шума за пики.

    Это грубый поиск (точность — шаг по x); точный центр даёт fit_peaks.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    threshold = max(prominence * (y.max() - y.min()), min_snr * noise_level(y))
    idx, _ = _scipy_find_peaks(y, prominence=threshold)
    return x[idx]


# ---------- аппроксимация ----------

def _model(x, params, n_peaks, shape):
    """Сумма пиков + прямая фона. params = [p1..., p2..., ..., a, b]."""
    k = _N_PARAMS[shape]
    total = params[-2] + params[-1] * x  # фон a + b·x
    for i in range(n_peaks):
        total = total + _PROFILES[shape](x, *params[i * k:(i + 1) * k])
    return total


def fit_peaks(x, y, centers, shape="lorentz", margin=30.0):
    """Аппроксимирует пики около заданных центров одновременно.

    x, y     — спектр (лучше несглаженный);
    centers  — примерные положения пиков, например из find_peaks;
    shape    — "lorentz", "gauss" или "voigt" (псевдо-Фойгт);
    margin   — сколько см⁻¹ захватить слева и справа от крайних пиков.

    Перекрывающиеся пики (например, TO и LO у GaAs) подгоняются вместе.
    Под пиками подгоняется линейный фон.

    Возвращает (table, fit):
      table — pandas.DataFrame: center, fwhm, amplitude, area и их
              погрешности *_err (1σ из ковариационной матрицы подгонки);
      fit   — словарь для графиков: x, y, model, baseline, components, r2.

    Погрешности — статистические: они не учитывают калибровку прибора
    и выбор формы пика. FWHM включает аппаратное уширение спектрометра.
    """
    if shape not in _PROFILES:
        raise ValueError(f"shape должен быть одним из {list(_PROFILES)}")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    centers = np.sort(np.atleast_1d(np.asarray(centers, dtype=float)))
    n = len(centers)
    k = _N_PARAMS[shape]

    # 1. Вырезаем участок вокруг пиков
    mask = (x >= centers[0] - margin) & (x <= centers[-1] + margin)
    xs, ys = x[mask], y[mask]
    n_params = n * k + 2
    if len(xs) <= n_params:
        raise ValueError(
            f"В участке всего {len(xs)} точек, а параметров {n_params}: "
            "увеличьте margin или уменьшите число пиков"
        )
    step = np.median(np.diff(xs))

    # 2. Начальные приближения
    # Фон — прямая через края участка
    b0 = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    a0 = ys[0] - b0 * xs[0]
    p0, lower, upper = [], [], []
    span = ys.max() - ys.min()
    for c in centers:
        # Самая высокая точка рядом с заданным центром
        near = np.flatnonzero(np.abs(xs - c) <= 2 * step)
        if len(near) == 0:
            near = [np.argmin(np.abs(xs - c))]
        i = near[np.argmax(ys[near])]
        height = max(ys[i] - (a0 + b0 * xs[i]), span * 0.01)
        # Ширина на полувысоте по данным; если не получилось — 3 шага
        try:
            with warnings.catch_warnings():  # предупреждение не нужно: ниже есть запасной вариант
                warnings.simplefilter("ignore")
                w = peak_widths(ys, [i], rel_height=0.5)[0][0] * step
        except ValueError:
            w = 0.0
        if not np.isfinite(w) or w <= 0:
            w = 3 * step
        shift = max(w, 2 * step)  # насколько центр может уйти от начального
        p0 += [c, w, height]
        # FWHM не меньше полшага: более узкий пик по таким точкам не различить,
        # а без этой границы подгонка слабого пика «схлопывается» в иглу
        lower += [c - shift, step * 0.5, 0.0]
        upper += [c + shift, (xs[-1] - xs[0]), np.inf]
        if shape == "voigt":
            p0 += [0.5]
            lower += [0.0]
            upper += [1.0]
    p0 += [a0, b0]
    lower += [-np.inf, -np.inf]
    upper += [np.inf, np.inf]

    # 3. Подгонка методом наименьших квадратов.
    # *params — «распаковка»: превращает список в отдельные аргументы функции
    def f(xx, *params):
        return _model(xx, np.array(params), n, shape)

    try:
        popt, pcov = curve_fit(f, xs, ys, p0=p0, bounds=(lower, upper), maxfev=5000)
    except RuntimeError as error:  # try/except — «перехват» ошибки
        raise ValueError(
            "Подгонка не сошлась: попробуйте другие центры, margin или форму пика"
        ) from error
    perr = np.sqrt(np.diag(pcov))

    # 4. Таблица результатов
    rows = []
    for i in range(n):
        p = popt[i * k:(i + 1) * k]
        eta = p[3] if shape == "voigt" else (1.0 if shape == "lorentz" else 0.0)
        area = peak_area(p[1], p[2], shape, eta)
        # Погрешность площади: переносим ковариацию через численные производные
        sl = slice(i * k, (i + 1) * k)
        grad = np.zeros(k)
        for j in range(k):
            dp = np.array(p, dtype=float)
            h = 1e-6 * max(abs(dp[j]), 1e-6)
            dp[j] += h
            e = dp[3] if shape == "voigt" else eta
            grad[j] = (peak_area(dp[1], dp[2], shape, e) - area) / h
        area_err = float(np.sqrt(grad @ pcov[sl, sl] @ grad))
        row = {
            "center": p[0], "center_err": perr[i * k],
            "fwhm": p[1], "fwhm_err": perr[i * k + 1],
            "amplitude": p[2], "amplitude_err": perr[i * k + 2],
            "area": area, "area_err": area_err,
        }
        if shape == "voigt":
            row["eta"] = p[3]
            row["eta_err"] = perr[i * k + 3]
        row["shape"] = shape
        rows.append(row)
    table = pd.DataFrame(rows)

    # 5. Данные для графика
    model = _model(xs, popt, n, shape)
    baseline = popt[-2] + popt[-1] * xs
    components = [_PROFILES[shape](xs, *popt[i * k:(i + 1) * k]) for i in range(n)]
    ss_res = np.sum((ys - model) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    fit = {
        "x": xs, "y": ys, "model": model, "baseline": baseline,
        "components": components, "r2": 1 - ss_res / ss_tot,
        "baseline_coef": (popt[-2], popt[-1]),
    }
    return table, fit


def curves_on_grid(table, fit, x):
    """Считает фон, отдельные пики и сумму на любой сетке x — для гладких графиков.

    При грубом шаге (например, 6 см⁻¹) кривая по точкам данных выглядит
    ломаной; на густой сетке видна настоящая форма подогнанных пиков.
    Возвращает (baseline, components, total).
    """
    x = np.asarray(x, dtype=float)
    a, b = fit["baseline_coef"]
    baseline = a + b * x
    components = []
    for _, r in table.iterrows():  # iterrows() перебирает строки таблицы
        args = [r.center, r.fwhm, r.amplitude] + ([r.eta] if r["shape"] == "voigt" else [])
        components.append(_PROFILES[r["shape"]](x, *args))
    total = baseline + sum(components)
    return baseline, components, total
