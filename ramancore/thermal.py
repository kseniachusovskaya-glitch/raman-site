"""Температура по сдвигу фононного пика (рамановская термометрия).

Идея: при нагреве решётка расширяется и колебания становятся
ангармоничными — фононный пик сдвигается в сторону меньших частот.
Сначала по серии спектров при известных температурах строится
калибровка ω(T), затем по положению пика в новом спектре находится T
(например, нагрев образца лазером).

Модель — ангармонизм трёх- и четырёхфононных процессов:
    ω(T) = ω0 + A·[1 + 2/(eˣ − 1)] + B·[1 + 3/(eʸ − 1) + 3/(eʸ − 1)²],
    x = h·c·ω0 / (2·k·T),  y = h·c·ω0 / (3·k·T)
Balkanski, Wallis, Haro, Phys. Rev. B 28, 1928 (1983); так же в статье
Максимов, Тарасов, Санжаровский, Чусовская, Князев, Квантовая электроника
56(4), 247 (2026), где для LO i-GaAs получено ω0 = 302.0, A = −0.93,
B = −0.11 см⁻¹ (вместе с отдельным членом теплового расширения).
Здесь ω0, A, B — подгоночные параметры, свои для каждого материала,
моды и прибора; калибровать нужно на своём приборе. Член B по умолчанию
выключен: при 5–6 точках калибровки он плохо отделяется от A.
"""

import numpy as np
from scipy.optimize import brentq, curve_fit

# h·c/k, см·К (вторая радиационная постоянная, CODATA 2018)
_HC_K = 1.438776877


def balkanski(T, omega0, A, B=0.0):
    """Положение пика при температуре T (К) по модели Балканского."""
    T = np.asarray(T, dtype=float)
    ex = np.expm1(_HC_K * omega0 / (2 * T))  # expm1(x) = exp(x) − 1, точнее при малых x
    ey = np.expm1(_HC_K * omega0 / (3 * T))
    return omega0 + A * (1 + 2 / ex) + B * (1 + 3 / ey + 3 / ey**2)


def fit_temperature_shift(T, omega, omega_err=None, four_phonon=False):
    """Калибровка: подгоняет ω(T) по серии спектров при известных температурах.

    T         — температуры, К (не меньше 3 точек);
    omega     — положения пика, см⁻¹ (например, center из fit_peaks);
    omega_err — их погрешности (center_err); если есть, точки взвешиваются;
    four_phonon — подгонять и член B (нужно больше точек и широкий диапазон T).

    Возвращает словарь: omega0, A, B, их погрешности, T_range (диапазон
    калибровки) и slope — средний наклон dω/dT в этом диапазоне, см⁻¹/К.
    """
    T, omega = np.asarray(T, dtype=float), np.asarray(omega, dtype=float)
    n_params = 3 if four_phonon else 2
    if len(T) < n_params + 1:
        raise ValueError(f"Для калибровки нужно хотя бы {n_params + 1} температуры")
    sigma = None if omega_err is None else np.maximum(np.asarray(omega_err, float), 1e-3)
    p0 = [omega.max() + 1.0, -1.0, -0.1][:n_params]  # ω0 чуть выше, A и B < 0
    popt, pcov = curve_fit(balkanski, T, omega, p0=p0, sigma=sigma,
                           absolute_sigma=sigma is not None, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    popt, perr = np.append(popt, [0.0] * (3 - n_params)), np.append(perr, [0.0] * (3 - n_params))
    lo, hi = T.min(), T.max()
    return {
        "omega0": popt[0], "A": popt[1], "B": popt[2],
        "omega0_err": perr[0], "A_err": perr[1], "B_err": perr[2],
        "T_range": (lo, hi),
        "slope": (balkanski(hi, *popt) - balkanski(lo, *popt)) / (hi - lo),
    }


def temperature_from_shift(omega, calibration, omega_err=0.0):
    """Температура по положению пика с помощью калибровки fit_temperature_shift.

    Возвращает (T, T_err, warnings). T_err — из погрешности положения пика
    через наклон dω/dT; погрешность самой калибровки не учитывается.
    """
    w0, A, B = calibration["omega0"], calibration["A"], calibration.get("B", 0.0)

    def f(T):
        return balkanski(T, w0, A, B) - omega

    # brentq ищет корень на отрезке, где функция меняет знак
    try:
        T = brentq(f, 5.0, 3000.0)
    except ValueError:
        return np.nan, np.nan, ["Положение пика вне модели (выше ω0 или нагрев > 3000 K)."]
    h = 0.5
    slope = (balkanski(T + h, w0, A, B) - balkanski(T - h, w0, A, B)) / (2 * h)
    T_err = abs(omega_err / slope)
    warnings = []
    lo, hi = calibration["T_range"]
    if not lo - 10 <= T <= hi + 10:
        warnings.append(f"T = {T:.0f} K вне диапазона калибровки {lo:.0f}–{hi:.0f} K: "
                        "это экстраполяция.")
    warnings.append("Сдвиг пика дают и напряжения, и легирование: калибровку нужно "
                    "делать на том же образце или таком же материале.")
    return T, T_err, warnings
