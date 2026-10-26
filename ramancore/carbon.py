"""Углеродные материалы: аморфный углерод и DLC (a-C, a-C:H, ta-C).

Спектр DLC в видимом свете — две широкие полосы:
  G (~1500–1600 см⁻¹) — растяжение связей sp² (и в кольцах, и в цепочках);
  D (~1350 см⁻¹)      — «дыхание» шестичленных колец sp², есть только
                        при наличии колец.
Подгонка по Ferrari & Robertson, Phys. Rev. B 61, 14095 (2000):
D — лоренциан, G — несимметричный профиль Брейта–Вигнера–Фано (BWF).
Под полосами — линейный фон: у a-C:H это хвост фотолюминесценции, и его
наклон даёт содержание водорода (Casiraghi et al., Phys. Rev. B 72,
085401 (2005)).
"""

import numpy as np
from scipy.optimize import curve_fit

from ramancore.peaks import lorentzian


def bwf(x, center, fwhm, amplitude, q):
    """Профиль Брейта–Вигнера–Фано (Ferrari & Robertson 2000):

        I(ω) = I0 · [1 + 2(ω − ω0)/(Q·Γ)]² / [1 + (2(ω − ω0)/Γ)²]

    center = ω0, fwhm = Γ, amplitude = I0, q = Q — параметр связи.
    При 1/Q → 0 переходит в лоренциан. Q < 0 — хвост в сторону меньших ω,
    как у G-полосы DLC. Максимум сдвинут от ω0: см. bwf_maximum.
    """
    s = 2 * (x - center) / fwhm
    return amplitude * (1 + s / q) ** 2 / (1 + s**2)


def bwf_maximum(center, fwhm, q):
    """Положение и высота максимума BWF (из dI/dω = 0):
    ω_max = ω0 + Γ/(2Q),  I_max = I0·(1 + 1/Q²)."""
    return center + fwhm / (2 * q), 1 + 1 / q**2


def _model(x, d_c, d_w, d_a, g_c, g_w, g_a, q, a, b):
    """D (лоренциан) + G (BWF) + линейный фон a + b·(x − 1000)."""
    return lorentzian(x, d_c, d_w, d_a) + bwf(x, g_c, g_w, g_a, q) + a + b * (x - 1000)


def fit_dlc(x, y, window=(800.0, 2000.0)):
    """Подгоняет D + G спектра аморфного углерода (исходного, без вычитания фона).

    window — диапазон подгонки, см⁻¹. Фон считается линейным: так его
    наклон m остаётся в результате (нужен для водорода).

    Возвращает словарь:
      pos_g, pos_g_err — положение МАКСИМУМА G, см⁻¹ (так его приводят
                         в статьях; ω0 BWF лежит выше);
      fwhm_g, fwhm_d, pos_d — ширины и положение D;
      id_ig, id_ig_err — отношение ВЫСОТ I(D)/I(G) (как у Ferrari &
                         Robertson 2000);
      q                — параметр связи BWF;
      m_ig             — наклон фона / высота G, мкм (для водорода);
      fit              — x, y, model, d, g, baseline для графика;
      warnings.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= window[0]) & (x <= window[1])
    xs, ys = x[keep], y[keep]
    if xs.size < 20:
        raise ValueError(f"В диапазоне {window[0]:.0f}–{window[1]:.0f} см⁻¹ слишком мало точек")

    # Начальные значения: фон — прямая по краям окна, полосы — по высоте
    edge = max(3, xs.size // 20)
    x_edges = np.r_[xs[:edge], xs[-edge:]]
    b0, a0 = np.polyfit(x_edges - 1000, np.r_[ys[:edge], ys[-edge:]], 1)
    rest = ys - (a0 + b0 * (xs - 1000))
    h = max(rest.max(), 1e-12)
    p0 = [1350, 250, 0.5 * h, 1560, 150, h, -5.0, a0, b0]
    lower = [1250, 30, 0, 1450, 30, 0, -50, -np.inf, -np.inf]
    upper = [1450, 500, np.inf, 1650, 400, np.inf, -0.5, np.inf, np.inf]
    popt, pcov = curve_fit(_model, xs, ys, p0=p0, bounds=(lower, upper), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    d_c, d_w, d_a, g_c, g_w, g_a, q, a, b = popt

    pos_g, k = bwf_maximum(g_c, g_w, q)
    i_g = g_a * k  # высота G в максимуме
    id_ig = d_a / i_g
    # Погрешности: положение максимума — от ω0 (вклад Γ/2Q меньше);
    # отношение высот — из относительных ошибок амплитуд
    id_ig_err = id_ig * np.hypot(perr[2] / max(d_a, 1e-12), perr[5] / max(g_a, 1e-12))
    m_ig = b / i_g * 1e4  # (1/см⁻¹) = см → мкм

    warnings = []
    if d_a < 3 * perr[2]:
        warnings.append("D почти не видна: I(D)/I(G) близко к нулю и определено грубо.")
    if q < -45:
        warnings.append("Q у границы подгонки: G почти симметрична (лоренциан).")
    return {
        "pos_g": pos_g, "pos_g_err": perr[3], "fwhm_g": g_w, "fwhm_g_err": perr[4],
        "pos_d": d_c, "pos_d_err": perr[0], "fwhm_d": d_w, "fwhm_d_err": perr[1],
        "id_ig": id_ig, "id_ig_err": id_ig_err, "q": q, "m_ig": m_ig,
        "fit": {"x": xs, "y": ys, "model": _model(xs, *popt),
                "d": lorentzian(xs, d_c, d_w, d_a), "g": bwf(xs, g_c, g_w, g_a, q),
                "baseline": a + b * (xs - 1000)},
        "warnings": warnings,
    }


def hydrogen_content(m_ig, laser_nm=None):
    """Содержание водорода в a-C:H, ат. %, по наклону фона фотолюминесценции:

        H = 21.7 + 16.6 · log10(m / I(G) [мкм])

    Casiraghi, Ferrari, Robertson, Phys. Rev. B 72, 085401 (2005); формула
    получена при возбуждении 514 нм и работает примерно от H ≈ 20 ат. %:
    при меньшем H фотолюминесценцию гасят дефекты, наклон почти нулевой.

    Возвращает (H, warnings); H = nan, если наклон не положительный.
    """
    warnings = []
    if laser_nm is not None and abs(laser_nm - 514) > 25:
        warnings.append(f"Формула получена для 514 нм, а спектр снят при {laser_nm} нм: "
                        "фотолюминесценция зависит от длины волны, оценка неверна.")
    if not m_ig > 0:
        return np.nan, warnings + ["Наклон фона не положительный: водорода меньше ~20 ат. % "
                                   "или фон искажён (подложка, вычитание фона)."]
    h = 21.7 + 16.6 * np.log10(m_ig)
    if h < 20:
        warnings.append("H < 20 ат. %: ниже области, где формула проверена.")
    warnings.append("Фон должен быть исходным: не вычитайте базовую линию перед расчётом.")
    return h, warnings


def g_dispersion(pos_g_1, laser_1, pos_g_2, laser_2):
    """Дисперсия G: Disp(G) = |ΔPos(G) / Δλ|, см⁻¹/нм, по спектрам на двух лазерах.

    У графита и нанокристаллического графита G почти не смещается с
    длиной волны (Disp(G) ≈ 0), у аморфного углерода смещается тем
    сильнее, чем больше беспорядок и доля sp³ (Ferrari & Robertson,
    Phys. Rev. B 64, 075414 (2001)). Знак: в аморфном углероде G выше
    при более короткой длине волны.
    """
    if laser_1 == laser_2:
        raise ValueError("Нужны спектры на двух разных лазерах")
    return abs((pos_g_2 - pos_g_1) / (laser_2 - laser_1))


# ---------- Графен: деформация и легирование по G и 2D ----------

# Lee, Ahn, Shim, Lee, Ryu, Nat. Commun. 3, 1024 (2012), doi:10.1038/ncomms2022
# (открытый доступ). Все числа — для возбуждения 514.5 нм:
#   (ωG0, ω2D0) = (1581.6, 2676.9) см⁻¹ — свободный графен без деформации и легирования;
#   Δω2D/ΔωG = 2.2 ± 0.2 — линия деформации (среднее по 8 образцам);
#   Δω2D/ΔωG = 0.70 ± 0.05 — линия дырочного легирования;
#   ΔωG = −69.1 ± 3.4 см⁻¹/% при двуосной деформации, −23.5 см⁻¹/% при одноосной.
GRAPHENE_REF_514 = (1581.6, 2676.9)
GRAPHENE_SLOPE_STRAIN = 2.2
GRAPHENE_SLOPE_DOPING = 0.70
GRAPHENE_G_PER_STRAIN = {"biaxial": -69.1, "uniaxial": -23.5}


def fit_graphene(x, y):
    """Подгоняет G (~1580) и 2D (~2700) лоренцианами, D (~1350) — если есть.

    Возвращает словарь: pos_g, fwhm_g, pos_2d, fwhm_2d, i2d_ig (по высотам),
    id_ig (по высотам; 0, если D не видна), tables — таблицы fit_peaks,
    fits — данные для графиков.
    """
    from ramancore.peaks import find_peaks, fit_peaks  # локально: нет цикла импортов

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    result, tables, fits = {}, {}, {}
    for name, guess, half in [("g", 1585, 60), ("2d", 2690, 120), ("d", 1350, 60)]:
        near = (x > guess - half) & (x < guess + half)
        if near.sum() < 8:
            continue
        found = find_peaks(x[near], y[near], min_snr=5)
        if not len(found):
            continue
        center = found[np.argmin(np.abs(found - guess))]
        table, fit = fit_peaks(x, y, [center], shape="lorentz", margin=half)
        # Пик должен быть внутри окна и уверенно выше шума: иначе это хвост
        # соседней полосы (например, D на склоне G)
        if (abs(table.center[0] - guess) > half - 10
                or table.amplitude[0] < 3 * table.amplitude_err[0]):
            continue
        tables[name], fits[name] = table, fit
        result[f"pos_{name}"] = table.center[0]
        result[f"pos_{name}_err"] = table.center_err[0]
        result[f"fwhm_{name}"] = table.fwhm[0]
        result[f"height_{name}"] = table.amplitude[0]
    if "pos_g" not in result:
        raise ValueError("Полоса G (~1580 см⁻¹) не найдена")
    result["i2d_ig"] = result["height_2d"] / result["height_g"] if "height_2d" in result else np.nan
    result["id_ig"] = result.get("height_d", 0.0) / result["height_g"]
    result["tables"], result["fits"] = tables, fits
    return result


def graphene_strain_doping(pos_g, pos_2d, reference=GRAPHENE_REF_514, strain="biaxial"):
    """Разделяет сдвиг (G, 2D) на вклад деформации и легирования (Lee et al. 2012).

    Сдвиг точки от эталона раскладывается по двум направлениям:
        (ΔωG, Δω2D) = a·(1, 2.2) + b·(1, 0.70),
    a — сдвиг G от деформации, b — от дырочного легирования.
    Деформация, % = a / (−69.1) для двуосной или a / (−23.5) для одноосной.

    reference — (ωG0, ω2D0) недеформированного нелегированного графена.
    По умолчанию — значения Lee et al. для 514.5 нм; 2D сильно зависит от
    длины волны лазера, поэтому для 532 нм и других лучше свой эталон.

    Возвращает словарь: dg_strain, dg_doping (см⁻¹), strain_percent, warnings.
    """
    dg, d2d = pos_g - reference[0], pos_2d - reference[1]
    # Решаем систему 2×2: [[1, 1], [s_T, s_H]] · [a, b] = [ΔωG, Δω2D]
    a, b = np.linalg.solve([[1, 1], [GRAPHENE_SLOPE_STRAIN, GRAPHENE_SLOPE_DOPING]], [dg, d2d])
    warnings = ["Модель для дырочного (p) легирования; электронное легирование "
                "смещает 2D по-другому.",
                "Эталон (ωG0, ω2D0) по умолчанию — для 514.5 нм: при другом лазере 2D "
                "смещается, задайте свой эталон."]
    if b < 0:
        warnings.append("Вклад легирования получился отрицательным: точка вне области "
                        "модели (n-легирование или неверный эталон).")
    return {"dg_strain": a, "dg_doping": b,
            "strain_percent": a / GRAPHENE_G_PER_STRAIN[strain], "warnings": warnings}
