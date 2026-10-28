"""Расчёты для полимеров. Пока — фазовый состав ПВДФ и его сополимеров.

У ПВДФ (–CH2–CF2–)n три основные кристаллические фазы:
  α — TGTG′, неполярная (из расплава);
  β — TTTT (плоский зигзаг), полярная: пьезо- и сегнетоэлектрическая;
  γ — T3GT3G′, слабополярная.
Фазы различают по маятниковым колебаниям CH2 в области 790–850 см⁻¹:
  α ≈ 795, γ ≈ 811, β ≈ 839 см⁻¹,
  и по ножничным CF2: α ≈ 610, β ≈ 510 см⁻¹.
Источники положений полос:
  Kobayashi, Tashiro, Tadokoro, Macromolecules 8, 158 (1975) — отнесение
      полос трёх форм (форма I = β, II = α, III = γ);
  Constantino et al., Appl. Spectrosc. 59, 275 (2005),
      doi:10.1366/0003702053585336 — 794 (α), 839 (β), табл. I;
  Barnakov et al., Opt. Mater. Express 8, 2579 (2018),
      doi:10.1364/OME.8.002579 — табл. 1: 485 γ, 513 β, 610 α, 795 α,
      812 γ, 840 β, 881 α+β;
  Деримедведь и др., Опт. и спектр. 133(11), 1132 (2025),
      doi:10.61011/OS.2025.11.62157.7942-25 — 797 α, 811 γ, 840 β и
      доли фаз по отношению интенсивностей этих линий; у сополимера с
      тетрафторэтиленом (ТФЭ) — слабые полосы ТФЭ 826–829 см⁻¹;
  Chapron et al., J. Raman Spectrosc. 52, 1073 (2021), doi:10.1002/jrs.6081 —
      модель с аморфной фазой: лоренцианы кристалла 795 и 874 (α), 839 (β),
      гауссианы аморфной фазы 797, 809, 830, 881 см⁻¹ (центры фиксированы),
      F(α) = ΣA(α)/(ΣA(α) + A(β)) по площадям лоренцианов, ур. (1);
  Purushothaman et al., ACS Appl. Polym. Mater. 6, 8291 (2024),
      doi:10.1021/acsapm.4c01157 — аморфная фаза по спектру расплава при
      200 °C (8 гауссианов 802, 812, 824, 836, 857, 876, 889, 904 см⁻¹,
      площади 8:4:4:2:1:2:4:1; лазер 785 нм, Kaiser RXN1), кристалл —
      Pearson VII: α 799 и 877, β 842, γ 814 и ≈830, общая α+β+γ 881;
      TC и F(α), F(β), F(γ) — ур. (10)–(13).

Три метода: fit_pvdf (Деримедведь 2025: α, γ, β со свободными центрами),
fit_pvdf_amorphous (Chapron 2021: α и β кристалла + аморфная фаза) и
fit_pvdf_melt (Purushothaman 2024: α, β, γ + аморфная фаза по расплаву).

Доли фаз в fit_pvdf — ОЦЕНКА по методу Деримедведь 2025: интенсивность линии
каждой фазы делится на сумму трёх линий. Это предполагает одинаковое
сечение рассеяния у всех трёх полос; калибровки по ИК или рентгену нет,
поэтому проценты сравнимы между образцами одной серии, но не абсолютны.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammaln

from ramancore.peaks import _GAUSS_AREA, fit_peaks, gaussian, lorentzian, peak_area

# Положения полос CH2 rocking, см⁻¹ (начальные значения для подгонки)
PVDF_BANDS = {"alpha": 795.0, "gamma": 811.0, "beta": 839.0}
PVDF_CC = 881.0     # νs(CC), есть у α и β — подгоняется, чтобы не мешала 839
TFE_BAND = 827.0    # полоса ТФЭ в сополимере ВДФ-ТФЭ (Деримедведь 2025: 826–829)
# Ножничные CF2 — вторая, независимая проверка α/β (Barnakov 2018, табл. 1)
PVDF_CF2 = {"beta": 511.0, "alpha": 610.0}
PHASE_NAMES = {"alpha": "α", "gamma": "γ", "beta": "β", "cc": "νs(CC), α+β", "tfe": "ТФЭ"}


def fit_pvdf(x, y, gamma=True, tfe=False, shape="voigt", window=(760.0, 910.0)):
    """Подгоняет полосы α (795), γ (811), β (839) и νs(CC) (881) ПВДФ.

    gamma — подгонять γ (811); выключите, если γ-фазы заведомо нет и
            подгонка «изобретает» пик;
    tfe   — сополимер с ТФЭ: добавить полосу 827 см⁻¹;
    shape — форма пиков; по умолчанию псевдо-Фойгт (смесь Гаусса и
            Лоренца, как у Constantino 2005).

    Возвращает словарь:
      table      — пики (center, fwhm, amplitude, area, *_err) с индексом
                   α, γ, β, νs(CC), ТФЭ;
      fit        — данные для графика (как у fit_peaks);
      fractions  — таблица долей фаз (%) по площади и по высоте;
      beta_alpha — I(β)/I(α) по площади и её погрешность.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= window[0]) & (x <= window[1])
    if keep.sum() < 15:
        raise ValueError(f"В области {window[0]:.0f}–{window[1]:.0f} см⁻¹ мало точек")
    names = ["alpha"] + (["gamma"] if gamma else []) + (["tfe"] if tfe else []) + ["beta", "cc"]
    centers = {"alpha": PVDF_BANDS["alpha"], "gamma": PVDF_BANDS["gamma"],
               "beta": PVDF_BANDS["beta"], "cc": PVDF_CC, "tfe": TFE_BAND}
    margin = max(min(centers[n] for n in names) - window[0], window[1] - max(centers[n] for n in names))
    table, fit = fit_peaks(x[keep], y[keep], [centers[n] for n in names], shape=shape,
                           margin=margin)
    # fit_peaks сортирует центры по возрастанию — так же отсортированы names
    names = sorted(names, key=lambda n: centers[n])
    table.index = [PHASE_NAMES[n] for n in names]
    phases = [n for n in ("alpha", "gamma", "beta") if n in names]
    fractions = phase_fractions(table, [PHASE_NAMES[n] for n in phases])
    a, b = table.loc["α"], table.loc["β"]
    ratio = b["area"] / a["area"] if a["area"] > 0 else np.inf
    rel = np.sqrt((a["area_err"] / a["area"]) ** 2 + (b["area_err"] / b["area"]) ** 2) \
        if a["area"] > 0 and b["area"] > 0 else np.nan
    return {"table": table, "fit": fit, "fractions": fractions,
            "beta_alpha": ratio, "beta_alpha_err": abs(ratio) * rel}


def phase_fractions(table, phases):
    """Доли фаз, %: I_фазы / Σ I по линиям phases (Деримедведь 2025).
    Считается по площади и по высоте (amplitude) — если они сильно
    расходятся, у полос разная ширина и результату стоит доверять меньше."""
    out = {}
    for value in ("area", "amplitude"):
        values = table.loc[phases, value].clip(lower=0)
        total = values.sum()
        out[value] = 100 * values / total if total > 0 else values * np.nan
    return pd.DataFrame({"по площади, %": out["area"], "по высоте, %": out["amplitude"]})


def pvdf_cf2_ratio(x, y, shape="voigt", margin=25.0):
    """Вторая проверка: I(β, 511)/I(α, 610) по ножничным CF2 (площади).
    Возвращает (ratio, table). Полосы слабые — на шумном спектре это лишь
    подтверждение «какой фазы больше», а не число."""
    table, _ = fit_peaks(x, y, [PVDF_CF2["beta"], PVDF_CF2["alpha"]], shape=shape,
                         margin=margin)
    table.index = ["β (CF2, 511)", "α (CF2, 610)"]
    alpha = table["area"].iloc[1]
    return (table["area"].iloc[0] / alpha if alpha > 0 else np.inf), table


# Модель Chapron 2021: (центр, форма, к чему относится)
CHAPRON_BANDS = [
    (795.0, "lorentz", "α"), (874.0, "lorentz", "α"), (839.0, "lorentz", "β"),
    (797.0, "gauss", "аморф."), (809.0, "gauss", "аморф."), (830.0, "gauss", "аморф."),
    (881.0, "gauss", "аморф."),
]
# Границы ширин (FWHM, см⁻¹) — НАШЕ допущение: в статье их нет, но без них
# лоренциан 795 и гауссиан 797 неразличимы. Кристаллические полосы узкие,
# аморфные — широкие.
CRYSTAL_FWHM = (3.0, 20.0)
AMORPHOUS_FWHM = (15.0, 60.0)


def fit_pvdf_amorphous(x, y, window=(750.0, 900.0)):
    """Разложение 750–900 см⁻¹ по модели Chapron et al. 2021: кристаллические
    полосы — лоренцианы (α: 795, 874; β: 839), аморфные — гауссианы
    (797, 809, 830, 881). Центры фиксированы, подгоняются ширины и высоты
    + линейный фон.

    Возвращает словарь:
      table   — полосы: center, shape, phase, fwhm, amplitude, area;
      fit     — x, y, model, baseline, components (для графика), r2;
      f_alpha, f_beta — доли α и β в кристаллической части, % (ур. 1);
      f_cryst, f_amorphous — доля площади лоренцианов и гауссианов, %.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= window[0]) & (x <= window[1])
    xs, ys = x[keep], y[keep]
    n = len(CHAPRON_BANDS)
    if len(xs) <= 2 * n + 2:
        raise ValueError(f"В области {window[0]:.0f}–{window[1]:.0f} см⁻¹ мало точек")
    profiles = {"lorentz": lorentzian, "gauss": gaussian}

    def model(xx, *p):
        total = p[-2] + p[-1] * xx
        for i, (c, shape, _) in enumerate(CHAPRON_BANDS):
            total = total + profiles[shape](xx, c, p[2 * i], p[2 * i + 1])
        return total

    b0 = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    a0 = ys[0] - b0 * xs[0]
    height = max(ys.max() - (a0 + b0 * xs).max(), 1e-9)
    p0, lower, upper = [], [], []
    for c, shape, _ in CHAPRON_BANDS:
        lo, hi = CRYSTAL_FWHM if shape == "lorentz" else AMORPHOUS_FWHM
        p0 += [(lo + hi) / 2 if shape == "gauss" else 8.0, height / 3]
        lower += [lo, 0.0]
        upper += [hi, np.inf]
    p0 += [a0, b0]
    lower += [-np.inf, -np.inf]
    upper += [np.inf, np.inf]
    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=(lower, upper), maxfev=20000)
    except RuntimeError as error:
        raise ValueError("Подгонка по модели Chapron не сошлась") from error

    rows, components = [], []
    for i, (c, shape, phase) in enumerate(CHAPRON_BANDS):
        fwhm, amp = popt[2 * i], popt[2 * i + 1]
        rows.append({"center": c, "shape": shape, "phase": phase, "fwhm": fwhm,
                     "amplitude": amp, "area": peak_area(fwhm, amp, shape)})
        components.append(profiles[shape](xs, c, fwhm, amp))
    table = pd.DataFrame(rows)
    area = table.groupby("phase")["area"].sum()
    crystal = area.get("α", 0.0) + area.get("β", 0.0)
    total = crystal + area.get("аморф.", 0.0)
    fitted = model(xs, *popt)
    r2 = 1 - np.sum((ys - fitted) ** 2) / np.sum((ys - ys.mean()) ** 2)
    return {
        "table": table,
        "fit": {"x": xs, "y": ys, "model": fitted, "baseline": popt[-2] + popt[-1] * xs,
                "components": components, "r2": r2},
        "f_alpha": 100 * area.get("α", 0.0) / crystal if crystal > 0 else np.nan,
        "f_beta": 100 * area.get("β", 0.0) / crystal if crystal > 0 else np.nan,
        "f_cryst": 100 * crystal / total if total > 0 else np.nan,
        "f_amorphous": 100 * area.get("аморф.", 0.0) / total if total > 0 else np.nan,
    }


# ---------- Purushothaman et al. 2024 ----------

# Аморфная фаза = спектр расплава при 200 °C, разложенный на 8 гауссианов
# (Purushothaman 2024, раздел 3.3.2.2): центры, см⁻¹, и отношение площадей
MELT_CENTERS = (802.0, 812.0, 824.0, 836.0, 857.0, 876.0, 889.0, 904.0)
MELT_AREA_RATIO = (8.0, 4.0, 4.0, 2.0, 1.0, 2.0, 4.0, 1.0)
# Кристаллические полосы (там же, табл. 3 и рис. 6): центр, фаза
MELT_CRYSTAL_BANDS = [
    (799.0, "α"), (814.0, "γ"), (830.0, "γ"), (842.0, "β"), (877.0, "α"),
    (881.0, "α+β+γ"),
]
# НАШИ допущения (в статье чисел нет): центры кристаллических полос могут
# сдвигаться на ±4 см⁻¹; ширины (FWHM) кристалла 2–20, аморфных
# гауссианов 6–40 см⁻¹.
MELT_SHIFT = 4.0
MELT_CRYSTAL_FWHM = (2.0, 20.0)
MELT_AMORPHOUS_FWHM = (6.0, 40.0)


def pearson7(x, center, fwhm, amplitude, m):
    """Pearson VII: m = 1 — Лоренц, m → ∞ — Гаусс (как в Purushothaman 2024)."""
    k = 2 ** (1 / m) - 1
    return amplitude / (1 + 4 * k * ((x - center) / fwhm) ** 2) ** m


def pearson7_area(fwhm, amplitude, m):
    """Площадь пика Pearson VII (аналитически, через гамма-функцию)."""
    k = 2 ** (1 / m) - 1
    return amplitude * fwhm / 2 * np.sqrt(np.pi / k) * np.exp(gammaln(m - 0.5) - gammaln(m))


def melt_template(x_melt, y_melt, x, window=(780.0, 920.0)):
    """Готовит свой спектр расплава как профиль аморфной фазы: вырезает окно,
    вычитает прямую через края окна (линейный фон, как в статье), переносит
    на сетку x образца и нормирует на площадь 1."""
    x_melt, y_melt = np.asarray(x_melt, dtype=float), np.asarray(y_melt, dtype=float)
    order = np.argsort(x_melt)
    x_melt, y_melt = x_melt[order], y_melt[order]
    if x_melt.min() > window[0] or x_melt.max() < window[1]:
        raise ValueError(f"Спектр расплава должен покрывать {window[0]:.0f}–{window[1]:.0f} см⁻¹")
    ends = np.interp(window, x_melt, y_melt)
    profile = np.interp(x, x_melt, y_melt)
    profile = profile - (ends[0] + (ends[1] - ends[0]) * (x - window[0]) / (window[1] - window[0]))
    profile = np.clip(profile, 0, None)
    area = np.trapezoid(profile, x)
    if area <= 0:
        raise ValueError("В спектре расплава нет сигнала в окне")
    return profile / area


def fit_pvdf_melt(x, y, melt=None, window=(780.0, 920.0)):
    """Разложение 780–920 см⁻¹ по Purushothaman et al. 2024.

    Аморфная фаза:
      melt=None      — 8 гауссианов статьи с фиксированными центрами и
                       отношением площадей 8:4:4:2:1:2:4:1 (общий масштаб,
                       ширины подгоняются в пределах MELT_AMORPHOUS_FWHM);
      melt=(xm, ym)  — свой спектр расплава ПВДФ (лучше при 200 °C на том же
                       приборе): его форма берётся целиком, подгоняется
                       только масштаб.
    Кристалл — 6 полос Pearson VII (MELT_CRYSTAL_BANDS), фон — прямая.

    Возвращает словарь:
      table — полосы: center, phase, fwhm, amplitude, m, area;
      fit   — x, y, model, baseline, components, amorphous (кривая), r2;
      tc    — общая кристалличность, %, ур. (10): все кристаллические
              площади (включая общую 881) / все площади;
      f_alpha, f_beta, f_gamma — доли фаз в кристалле, %, ур. (11)–(13)
              (общая полоса 881 в них не входит).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= window[0]) & (x <= window[1])
    xs, ys = x[keep], y[keep]
    n_cr = len(MELT_CRYSTAL_BANDS)
    own = melt is not None
    n_am = 0 if own else len(MELT_CENTERS)
    n_params = 4 * n_cr + n_am + 3
    if len(xs) <= n_params:
        raise ValueError(f"В области {window[0]:.0f}–{window[1]:.0f} см⁻¹ {len(xs)} точек, а "
                         f"параметров модели {n_params}: нужен шаг спектра ≤ 3 см⁻¹")
    if own:
        template = melt_template(melt[0], melt[1], xs, window)
    ratio = np.asarray(MELT_AREA_RATIO) / np.sum(MELT_AREA_RATIO)

    def amorphous(xx, p):
        """Кривая аморфной фазы; p = [масштаб (площадь), ширины гауссианов...]."""
        if own:
            return p[0] * template
        total = np.zeros_like(xx)
        for c, share, fwhm in zip(MELT_CENTERS, ratio, p[1:]):
            # высота гауссиана с площадью p[0]·share
            total = total + gaussian(xx, c, fwhm, p[0] * share / (fwhm * _GAUSS_AREA))
        return total

    def model(xx, *p):
        total = p[-2] + p[-1] * xx
        for i in range(n_cr):
            total = total + pearson7(xx, *p[4 * i:4 * i + 4])
        return total + amorphous(xx, p[4 * n_cr:-2])

    b0 = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    a0 = ys[0] - b0 * xs[0]
    signal = np.clip(ys - (a0 + b0 * xs), 0, None)
    height = max(signal.max(), 1e-9)
    p0, lower, upper = [], [], []
    for c, _ in MELT_CRYSTAL_BANDS:
        p0 += [c, 6.0, max(np.interp(c, xs, signal) / 2, height / 50), 2.0]
        lower += [c - MELT_SHIFT, MELT_CRYSTAL_FWHM[0], 0.0, 1.0]
        upper += [c + MELT_SHIFT, MELT_CRYSTAL_FWHM[1], np.inf, 20.0]
    total_area = max(np.trapezoid(signal, xs), 1e-9)
    p0 += [total_area / 2]
    lower += [0.0]
    upper += [np.inf]
    if not own:
        p0 += [15.0] * n_am
        lower += [MELT_AMORPHOUS_FWHM[0]] * n_am
        upper += [MELT_AMORPHOUS_FWHM[1]] * n_am
    p0 += [a0, b0]
    lower += [-np.inf, -np.inf]
    upper += [np.inf, np.inf]
    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=(lower, upper), maxfev=40000)
    except RuntimeError as error:
        raise ValueError("Подгонка по модели Purushothaman не сошлась") from error

    rows, components = [], []
    for i, (_, phase) in enumerate(MELT_CRYSTAL_BANDS):
        c, fwhm, amp, m = popt[4 * i:4 * i + 4]
        rows.append({"center": c, "phase": phase, "fwhm": fwhm, "amplitude": amp, "m": m,
                     "area": pearson7_area(fwhm, amp, m)})
        components.append(pearson7(xs, c, fwhm, amp, m))
    am = popt[4 * n_cr:-2]
    if own:
        rows.append({"center": np.nan, "phase": "аморф. (свой расплав)", "fwhm": np.nan,
                     "amplitude": np.nan, "m": np.nan, "area": am[0]})
    else:
        for c, share, fwhm in zip(MELT_CENTERS, ratio, am[1:]):
            rows.append({"center": c, "phase": "аморф.", "fwhm": fwhm,
                         "amplitude": am[0] * share / (fwhm * _GAUSS_AREA), "m": np.nan,
                         "area": am[0] * share})
    table = pd.DataFrame(rows)
    area = table.groupby("phase")["area"].sum()
    phases = sum(area.get(p, 0.0) for p in ("α", "β", "γ"))
    crystal = phases + area.get("α+β+γ", 0.0)
    total = crystal + am[0]
    fitted = model(xs, *popt)
    r2 = 1 - np.sum((ys - fitted) ** 2) / np.sum((ys - ys.mean()) ** 2)
    share = lambda p: 100 * area.get(p, 0.0) / phases if phases > 0 else np.nan
    return {
        "table": table,
        "fit": {"x": xs, "y": ys, "model": fitted, "baseline": popt[-2] + popt[-1] * xs,
                "components": components, "amorphous": amorphous(xs, am), "r2": r2},
        "tc": 100 * crystal / total if total > 0 else np.nan,
        "f_alpha": share("α"), "f_beta": share("β"), "f_gamma": share("γ"),
    }
