"""Полупроводниковая аналитика: расчёты по положению и форме пиков.

Все литературные значения помечены источником. Где написано «сверить» —
значение взято по памяти или из вторичного источника и требует проверки
по первоисточнику.
"""

import numpy as np

from ramancore.peaks import find_peaks, fit_peaks, noise_level

# ---------- GaAs: ориентация поверхности по правилам отбора ----------

# Положения фононов GaAs при комнатной температуре, см⁻¹.
# TO 267.2 — Zardo et al., Phys. Rev. B 80, 245324 (2009), arXiv:0910.5266
# (объёмный эталон); LO ≈ 291–292 — ориентировочно, сверить.
GAAS_TO = 268.0
GAAS_LO = 291.5

# Правила отбора для решётки цинковой обманки при обратном рассеянии
# (Zardo et al. 2009, раздел 2; классический источник — Cardona,
# Light Scattering in Solids II, 1982 — сверить):
#   (100): LO разрешена, TO запрещена;
#   (110): TO разрешена, LO запрещена;
#   (111): разрешены обе.
# Пороги ниже — НЕ из литературы, а наше рабочее допущение: «запрещённая»
# мода никогда не равна нулю (апертура объектива, разориентация,
# дефекты), поэтому сравниваем отношение с запасом.
RATIO_100 = 0.2   # I(TO)/I(LO) ниже — похоже на (100)
RATIO_110 = 5.0   # I(TO)/I(LO) выше — похоже на (110)


def gaas_orientation(x, y, shape="lorentz", margin=40.0,
                     ratio_100=RATIO_100, ratio_110=RATIO_110):
    """Оценивает ориентацию поверхности GaAs по отношению I(TO)/I(LO).

    TO (~268) и LO (~292 см⁻¹) подгоняются вместе (fit_peaks, линейный фон),
    интенсивность — площадь пика. Подгонка идёт по исходному спектру.

    Возвращает словарь:
      ratio, ratio_err — I(TO)/I(LO) и его погрешность 1σ;
      orientation      — "(100)", "(110)", "(111) / смешанная" или
                         "не определена" (если подгонка не сошлась);
      table, fit       — результат fit_peaks (для таблицы и графика);
      warnings         — список оговорок, которые нужно показать пользователю.

    Важно: это оценка. У легированного GaAs (n или p) LO-фонон
    взаимодействует с плазмонами, и вместо LO появляются связанные моды
    L− и L+; мода L− может лежать близко к TO и имитировать её. Тогда
    отношение TO/LO уже не говорит об ориентации однозначно.
    """
    # Начальные центры: найденные пики рядом с ожидаемыми (±15 см⁻¹),
    # чтобы подгонка справилась и со сдвинутыми пиками
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    near = (x > GAAS_TO - 30) & (x < GAAS_LO + 30)
    found = find_peaks(x[near], y[near], min_snr=3) if near.sum() > 3 else np.array([])
    centers = []
    for expected in (GAAS_TO, GAAS_LO):
        close = found[np.abs(found - expected) < 15]
        centers.append(close[np.argmin(np.abs(close - expected))] if len(close) else expected)
    if abs(centers[1] - centers[0]) < 10:  # нашёлся один пик на оба места
        centers = [GAAS_TO, GAAS_LO]

    warnings = []
    try:
        table, fit = fit_peaks(x, y, centers, shape=shape, margin=margin)
    except ValueError as error:
        return {"ratio": np.nan, "ratio_err": np.nan, "orientation": "не определена",
                "table": None, "fit": None, "warnings": [str(error)]}
    to, lo = table.iloc[0], table.iloc[1]

    # Если ни TO, ни LO заметно не выступают над шумом — это не GaAs
    sigma = noise_level(y[near]) if near.sum() > 3 else noise_level(y)
    if max(to.amplitude, lo.amplitude) < 5 * sigma:
        return {"ratio": np.nan, "ratio_err": np.nan, "orientation": "не определена",
                "table": table, "fit": fit,
                "warnings": ["Пиков TO и LO GaAs не видно над шумом — похоже, это не GaAs."]}

    if abs(to.center - GAAS_TO) > 5 or abs(lo.center - GAAS_LO) > 5:
        warnings.append(
            f"Пики найдены при {to.center:.1f} и {lo.center:.1f} см⁻¹ — далеко от "
            f"ожидаемых {GAAS_TO} и {GAAS_LO}: возможны напряжения, нагрев, легирование "
            "или это не GaAs."
        )
    if lo.area <= 0:
        return {"ratio": np.inf, "ratio_err": np.nan, "orientation": "(110)",
                "table": table, "fit": fit,
                "warnings": warnings + ["LO не найдена: отношение не определено."]}

    ratio = to.area / lo.area
    # Погрешность отношения: относительные ошибки складываются в квадратах
    rel = np.hypot(to.area_err / max(to.area, 1e-12), lo.area_err / lo.area)
    ratio_err = ratio * rel

    if ratio < ratio_100:
        orientation = "(100)"
    elif ratio > ratio_110:
        orientation = "(110)"
    else:
        orientation = "(111) / смешанная"
        warnings.append(
            "Обе моды сравнимы: это (111), разориентированная пластина, "
            "поликристалл или вклад связанных плазмон-фононных мод."
        )
    if ratio_err > ratio:
        warnings.append(
            "Погрешность больше самого отношения: TO и LO не разрешены "
            "(шаг прибора слишком крупный или пик слишком слабый)."
        )
    if (ratio - ratio_err < ratio_100 < ratio + ratio_err) or \
       (ratio - ratio_err < ratio_110 < ratio + ratio_err):
        warnings.append("Отношение близко к порогу с учётом погрешности — вывод ненадёжен.")
    warnings.append(
        "Для легированного GaAs отношение TO/LO искажают плазмон-фононные моды (L−, L+)."
    )
    return {"ratio": ratio, "ratio_err": ratio_err, "orientation": orientation,
            "table": table, "fit": fit, "warnings": warnings}


# ---------- Концентрация электронов по связанной плазмон-фононной моде ----------

# Физические постоянные (CODATA 2018, точные или рекомендованные значения)
_E = 1.602176634e-19       # заряд электрона, Кл
_EPS0 = 8.8541878128e-12   # электрическая постоянная, Ф/м
_ME = 9.1093837015e-31     # масса электрона, кг
_C_CM = 2.99792458e10      # скорость света, см/с (перевод рад/с → см⁻¹)

# Параметры материалов для модели LO-фонон–плазмон.
#   TO, LO — частоты фононов нелегированного материала, см⁻¹;
#   eps_inf — оптическая (высокочастотная) диэлектрическая проницаемость;
#   m_eff — эффективная масса электрона на дне зоны Γ, в массах m0.
# Лучше всего передавать TO и LO, измеренные на своём нелегированном
# образце на том же приборе (так сделано в статьях Ксении): тогда
# сдвиг калибровки прибора не влияет на результат.
LOPC_PARAMS = {
    "GaAs": {
        # TO 33.2 мэВ и LO 36.1 мэВ (300 K) — ioffe.ru, NSM Archive, GaAs/optic;
        # ε∞ = n² = 3.299² по ИК-показателю преломления (там же, Blakemore 1982)
        "TO": 267.8, "LO": 291.2, "eps_inf": 10.88,
        # m* — Vurgaftman, Meyer, Ram-Mohan, J. Appl. Phys. 89, 5815 (2001)
        "m_eff": 0.067,
    },
    "InAs": {
        # TO, LO, ε∞ — СВЕРИТЬ по первоисточнику (пока справочные значения)
        "TO": 218.0, "LO": 243.0, "eps_inf": 12.25,
        "m_eff": 0.026,  # Vurgaftman et al. 2001
    },
    "InSb": {
        # TO, LO, ε∞ — СВЕРИТЬ по первоисточнику (пока справочные значения)
        "TO": 180.0, "LO": 191.0, "eps_inf": 15.68,
        "m_eff": 0.0135,  # Vurgaftman et al. 2001
    },
}


def _lopc_params(material, overrides):
    """Параметры материала с заменой тех, что передал пользователь."""
    if material not in LOPC_PARAMS:
        raise ValueError(f"Нет параметров для {material}; есть: {list(LOPC_PARAMS)}")
    params = dict(LOPC_PARAMS[material])
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def plasma_frequency(n, eps_inf, m_eff):
    """Плазменная частота электронов, см⁻¹.

    ωp² = n·e² / (ε0·ε∞·m*),  n — концентрация в см⁻³.
    Модель параболической зоны: m* не зависит от n.
    """
    n_m3 = np.asarray(n, dtype=float) * 1e6  # см⁻³ → м⁻³
    omega = np.sqrt(n_m3 * _E**2 / (_EPS0 * eps_inf * m_eff * _ME))  # рад/с
    return omega / (2 * np.pi * _C_CM)


def coupled_modes(n, material="GaAs", **overrides):
    """Частоты связанных мод L− и L+ при концентрации электронов n (см⁻³).

    2ω±² = (ωLO² + ωp²) ± √((ωLO² + ωp²)² − 4·ωp²·ωTO²)
    (без затухания; см., например, Yu & Cardona, Fundamentals of
    Semiconductors, Springer 2010).
    overrides — свои значения TO, LO, eps_inf, m_eff.

    Возвращает (L_minus, L_plus, omega_p) в см⁻¹.
    """
    p = _lopc_params(material, overrides)
    wp = plasma_frequency(n, p["eps_inf"], p["m_eff"])
    s = p["LO"]**2 + wp**2
    d = np.sqrt(s**2 - 4 * wp**2 * p["TO"]**2)
    return np.sqrt((s - d) / 2), np.sqrt((s + d) / 2), wp


def concentration_from_mode(omega, material="GaAs", omega_err=0.0, **overrides):
    """Обратная задача: концентрация электронов по частоте L+ или L− (см⁻¹).

    Из формулы мод: ωp² = ω²(ω² − ωLO²) / (ω² − ωTO²), затем
    n = ωp²·ε0·ε∞·m* / e². Подходит L+ (ω > LO) и L− (ω < TO);
    между TO и LO решения нет.

    Возвращает (n, n_err) в см⁻³; n_err — из погрешности частоты omega_err.
    """
    p = _lopc_params(material, overrides)

    def n_of(w):
        if p["TO"] <= w <= p["LO"]:
            raise ValueError(f"{w:.1f} см⁻¹ между TO и LO — это не связанная мода")
        wp2 = w**2 * (w**2 - p["LO"]**2) / (w**2 - p["TO"]**2)  # в (см⁻¹)²
        wp_rad = np.sqrt(wp2) * 2 * np.pi * _C_CM
        return wp_rad**2 * _EPS0 * p["eps_inf"] * p["m_eff"] * _ME / _E**2 / 1e6

    n = n_of(omega)
    # Погрешность — численной производной dn/dω
    h = 0.01
    n_err = abs(n_of(omega + h) - n_of(omega - h)) / (2 * h) * omega_err
    return n, n_err


def lplus_concentration(x, y, material="GaAs", window=None, shape="gauss", **overrides):
    """Находит широкую моду L+ в спектре n-типа и оценивает концентрацию электронов.

    window — диапазон поиска L+, см⁻¹; по умолчанию от LO + 20 до 1000.
    L+ обычно широкая (десятки см⁻¹) и слабая, поэтому подгоняется
    одним гауссианом с линейным фоном по слегка сглаженному спектру.

    Возвращает словарь: omega, omega_err, n, n_err, table, fit, warnings.
    """
    p = _lopc_params(material, overrides)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lo, hi = window or (p["LO"] + 20, 1000.0)
    inside = (x >= lo) & (x <= hi)
    empty = {"omega": np.nan, "omega_err": np.nan, "n": np.nan, "n_err": np.nan,
             "table": None, "fit": None}
    if inside.sum() < 10:
        return {**empty, "warnings": ["В спектре нет диапазона, где искать L+."]}

    # Начальная точка — максимум сглаженного спектра (скользящее среднее
    # по ~15 см⁻¹: L+ широкая, а шум узкий)
    step = np.median(np.diff(x))
    k = max(3, int(round(15 / step)))
    ys = np.convolve(y, np.ones(k) / k, mode="same")  # convolve — скользящее среднее
    xi, yi = x[inside][k:-k], ys[inside][k:-k]
    start = xi[np.argmax(yi)]
    try:
        table, fit = fit_peaks(x, ys, [start], shape=shape, margin=150)
    except ValueError as error:
        return {**empty, "warnings": [str(error)]}
    peak = table.iloc[0]
    sigma = noise_level(y[inside])
    warnings = []
    # Пик засчитывается, если он выше шума сглаженного спектра (σ/√k)
    # и его высота определена надёжно (больше 5 своих погрешностей)
    weak = peak.amplitude < 3 * sigma / np.sqrt(k) or peak.amplitude < 5 * peak.amplitude_err
    if weak or not (lo <= peak.center <= hi):
        # table и fit не возвращаем: на графике не должно быть «подгонки» шума
        return {**empty,
                "warnings": ["Широкой моды L+ не видно над шумом: концентрация "
                             "мала, образец p-типа или L+ слишком затухает."]}

    n, n_err = concentration_from_mode(peak.center, material, peak.center_err, **overrides)
    warnings += [
        "Модель параболической зоны: m* не зависит от n. При n > ~1e18 см⁻³ "
        "непараболичность увеличивает m*, и настоящая n будет выше.",
        "Погрешность учитывает только положение пика; ширина L+ и выбор "
        "TO/LO/ε∞/m* дают ещё десятки процентов.",
        "Для p-типа метод не работает: дырочные плазмоны сильно затухают.",
    ]
    if peak.fwhm > 200:
        warnings.append("L+ очень широкая (FWHM > 200 см⁻¹): положение ненадёжно.")
    return {"omega": peak.center, "omega_err": peak.center_err, "n": n, "n_err": n_err,
            "table": table, "fit": fit, "warnings": warnings}
