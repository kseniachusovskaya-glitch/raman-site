"""Полупроводниковая аналитика: расчёты по положению и форме пиков.

Все литературные значения помечены источником. Где написано «сверить» —
значение взято по памяти или из вторичного источника и требует проверки
по первоисточнику.
"""

import numpy as np

from ramancore.peaks import find_peaks, fit_peaks, intrinsic_fwhm, noise_level

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
    (без затухания; Yu & Cardona, Fundamentals of Semiconductors, Springer
    2010; та же формула с элементарным зарядом e в ωp — Максимов и др.,
    Квантовая электроника 56(4), 247 (2026)).
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


# ---------- Качество кристалла по TO и LO ----------

def phonon_quality(x, y, material="GaAs", instrument_fwhm=None, reference_lo=None,
                   margin=30.0):
    """Ширины, положения и доля LO для TO и LO решётки цинковой обманки.

    Подгоняет TO и LO вместе (псевдо-Фойгт, линейный фон) и считает:
      table     — center, fwhm, area с погрешностями (+ intrinsic_fwhm, если
                  задана instrument_fwhm — аппаратная ширина прибора, см⁻¹);
      lo_fraction, lo_fraction_err — S(LO) / (S(TO) + S(LO)) по площадям:
                  для (100) у совершенного кристалла близко к 1, при
                  разупорядочении падает (запрет на TO снимается);
      lo_shift  — сдвиг LO относительно reference_lo (LO эталонного
                  нелегированного образца на том же приборе), если задан.
    Ориентиры TO/LO берутся из LOPC_PARAMS[material]; пики ищутся в ±15 см⁻¹
    от них, поэтому сдвиг калибровки прибора на несколько см⁻¹ не мешает.

    Возвращает словарь: table, fit, lo_fraction, lo_fraction_err, lo_shift,
    warnings.
    """
    p = _lopc_params(material, {})
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    near = (x > p["TO"] - 30) & (x < p["LO"] + 30)
    found = find_peaks(x[near], y[near], min_snr=3) if near.sum() > 3 else np.array([])
    centers = []
    for expected in (p["TO"], p["LO"]):
        close = found[np.abs(found - expected) < 15]
        centers.append(close[np.argmin(np.abs(close - expected))] if len(close) else expected)
    if abs(centers[1] - centers[0]) < 10:  # один пик на оба места
        centers = [p["TO"], p["LO"]]
    try:
        table, fit = fit_peaks(x, y, centers, shape="voigt", margin=margin)
    except ValueError as error:
        return {"table": None, "fit": None, "lo_fraction": np.nan, "lo_fraction_err": np.nan,
                "lo_shift": np.nan, "warnings": [str(error)]}
    table.index = ["TO", "LO"]
    warnings = []
    if instrument_fwhm is not None:
        table["intrinsic_fwhm"] = intrinsic_fwhm(table.fwhm, instrument_fwhm)
        if (table.fwhm < 1.5 * instrument_fwhm).any():
            warnings.append("Ширина пика близка к аппаратной: собственная ширина "
                            "определена грубо (погрешность может быть > 50 %).")
    to, lo = table.loc["TO"], table.loc["LO"]
    total = to.area + lo.area
    frac = lo.area / total
    # Погрешность доли: d(f)/dS_LO = S_TO/total², d(f)/dS_TO = −S_LO/total²
    frac_err = np.hypot(to.area * lo.area_err, lo.area * to.area_err) / total**2
    shift = lo.center - reference_lo if reference_lo is not None else np.nan
    warnings += [
        "Доля LO зависит от ориентации: сравнивайте образцы одной ориентации, (100).",
        "У легированного n-типа в TO входит связанная мода L−, а LO идёт из "
        "обеднённого слоя: сравнивайте образцы с близкой концентрацией.",
    ]
    return {"table": table, "fit": fit, "lo_fraction": frac, "lo_fraction_err": frac_err,
            "lo_shift": shift, "warnings": warnings}


# ---------- Механические напряжения по сдвигу фонона ----------

# Si, пик ~520 см⁻¹, поверхность (100). Одноосное напряжение σ или
# сумма σxx + σyy при двуосном (плёнка, край окна):
#     σ (МПа) = −435 · Δω (см⁻¹)
# De Wolf, Spectroscopy Europe 15(2), 6 (2003); подробный вывод —
# De Wolf, Semicond. Sci. Technol. 11, 139 (1996).
# Знак: пик выше эталона (Δω > 0) — сжатие (σ < 0), ниже — растяжение.
SI_STRESS_MPA_PER_CM = -435.0

# Модовые параметры Грюнайзена γ = −d ln ω / d ln V (гидростатическое
# сжатие): Wickboldt, Anastassakis, Sauer, Cardona, Phys. Rev. B 35, 1362
# (1987), по таблице II в Eryigit & Herman, Phys. Rev. B 53, 7775 (1996).
# γ(LO) = 1.23 совпадает со статьёй Ксении (Квантовая электроника 2026).
GRUNEISEN = {
    "GaAs": {"TO": 1.39, "LO": 1.23},
}

# GaAs, двуосное напряжение в плоскости (001). Cerdeira, Buchenauer,
# Pollak, Cardona, Phys. Rev. B 5, 580 (1972), doi:10.1103/PhysRevB.5.580:
#   таблица I — упругие податливости, 10⁻¹² см²/дин:
#     S11 + 2·S12 = 0.445,  S11 − S12 = 1.54;
#   таблица II — (p − q)/2ω0² = 0.1 ± 0.1 (измерено на TO).
# Сумму p + 2q берём из параметра Грюнайзена: γ = −(p + 2q)/6ω0²
# (там же, определение под таблицей II). Те же p, q применяем и к LO —
# так делают авторы (расщепление LO–TO от напряжения не зависит).
GAAS_S11_PLUS_2S12 = 0.445e-2   # 1/ГПа (1 ГПа = 10¹⁰ дин/см²)
GAAS_S11_MINUS_S12 = 1.54e-2    # 1/ГПа
GAAS_P_MINUS_Q_HALF = (0.1, 0.1)  # (значение, погрешность), в ω0²


def si_stress(omega, omega0, omega_err=0.0):
    """Напряжение в Si по положению пика, МПа.

    omega  — измеренный центр пика Si, см⁻¹ (center из fit_peaks);
    omega0 — центр пика ненапряжённого Si на том же приборе и лазере
             (например, своя пластина-эталон): так сдвиг калибровки
             прибора сокращается;
    omega_err — погрешность omega.

    Возвращает (sigma, sigma_err). Для двуосного напряжения sigma —
    сумма σxx + σyy; для изотропного двуосного σxx = σyy = sigma / 2.
    """
    shift = np.asarray(omega, dtype=float) - omega0
    return SI_STRESS_MPA_PER_CM * shift, abs(SI_STRESS_MPA_PER_CM) * np.asarray(omega_err)


def volume_change(omega, omega0, material="GaAs", mode="LO", omega_err=0.0):
    """Относительное изменение объёма ячейки ΔV/V (в %) по сдвигу фонона.

    Из определения параметра Грюнайзена: Δω/ω0 = −γ·ΔV/V, то есть
    ΔV/V = −(ω − ω0) / (γ·ω0). Отрицательное — сжатие решётки.
    Строго верно для всестороннего (гидростатического) сжатия; при
    одноосном или двуосном напряжении это лишь оценка.

    Возвращает (dV, dV_err) в процентах.
    """
    if material not in GRUNEISEN or mode not in GRUNEISEN[material]:
        raise ValueError(f"Нет параметра Грюнайзена для {material} {mode}")
    gamma = GRUNEISEN[material][mode]
    shift = np.asarray(omega, dtype=float) - omega0
    return -100 * shift / (gamma * omega0), 100 * np.asarray(omega_err) / (gamma * omega0)


def gaas_biaxial_coefficient(omega0, mode="LO"):
    """Сдвиг фонона GaAs при двуосном напряжении в плоскости (001), см⁻¹/ГПа.

    Напряжение σxx = σyy = σ (плёнка, пластина на подложке), съёмка с
    поверхности (001). Деформации: εxx = εyy = (S11 + S12)·σ, εzz = 2·S12·σ.
    LO (и TO, видимая на (001)) — мода с вектором вдоль z, её сдвиг
    (уравнение (2) у Cerdeira et al. 1972):
        Δω = ω0/2 · [p'·εzz + q'·(εxx + εyy)],  p' = p/ω0², q' = q/ω0².

    Возвращает (b, b_err): Δω = b·σ. b < 0 — растяжение смягчает фонон.
    Погрешность — только от (p − q) из таблицы II (~10 %); неточность γ
    и перенос p, q с TO на LO в неё не входят.
    """
    gamma = GRUNEISEN["GaAs"][mode]
    s11 = (GAAS_S11_PLUS_2S12 + 2 * GAAS_S11_MINUS_S12) / 3
    s12 = (GAAS_S11_PLUS_2S12 - GAAS_S11_MINUS_S12) / 3
    exx, ezz = s11 + s12, 2 * s12  # деформации на 1 ГПа

    def b(pq_half):
        # p + 2q = −6γ, p − q = 2·pq_half  →  решаем систему для p и q
        q = (-6 * gamma - 2 * pq_half) / 3
        p = q + 2 * pq_half
        return omega0 / 2 * (p * ezz + q * 2 * exx)

    value, err = GAAS_P_MINUS_Q_HALF
    return b(value), abs(b(value + err) - b(value - err)) / 2


def gaas_biaxial_stress(omega, omega0, mode="LO", omega_err=0.0):
    """Двуосное напряжение σ в GaAs (001) по сдвигу LO или TO, ГПа.

    omega0 — фонон ненапряжённого GaAs на том же приборе в тот же день.
    σ > 0 — растяжение, σ < 0 — сжатие. Возвращает (sigma, sigma_err),
    где погрешность включает и неточность коэффициента.
    """
    b, b_err = gaas_biaxial_coefficient(omega0, mode)
    shift = np.asarray(omega, dtype=float) - omega0
    sigma = shift / b
    err = np.hypot(np.asarray(omega_err) / b, sigma * b_err / b)
    return sigma, err
