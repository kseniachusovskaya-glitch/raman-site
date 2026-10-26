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


# ---------- Графен: число слоёв по форме 2D ----------

# Ferrari et al., Phys. Rev. Lett. 97, 187401 (2006), doi:10.1103/PhysRevLett.97.187401.
# Монослой: одна узкая симметричная 2D (лоренциан), примерно в 4 раза выше G.
# Двухслойный (AB): 2D из четырёх компонент; Table I — их сдвиги, см⁻¹, от
# среднего положения двух главных (эксперимент):
GRAPHENE_BILAYER_SPLIT = {514.5: (-44, -10, 10, 25), 633: (-55, -10, 10, 30)}
# Графит: две компоненты 2D1 и 2D2, примерно 1/4 и 1/2 высоты G.
# Турбостратный графит: одна 2D, но FWHM ≈ 50 см⁻¹ — почти вдвое шире, чем у
# монослоя. Отсюда порог ширины монослоя — середина между ними (наш выбор):
GRAPHENE_MONOLAYER_MAX_FWHM = 35.0


def _lorentz_sum(x, centers, fwhm, amplitudes, a, b, x0):
    return sum(lorentzian(x, c, fwhm, h) for c, h in zip(centers, amplitudes)) + a + b * (x - x0)


def graphene_layers(x, y, pos_2d, height_g, laser_nm=532, half=150.0):
    """Оценка числа слоёв по форме 2D (Ferrari et al. 2006).

    2D подгоняется двумя моделями на одном линейном фоне:
      1) один лоренциан — монослой или турбостратный графит;
      2) четыре лоренциана с общей шириной и сдвигами из Table I — двухслойный.
    Если вторая модель описывает спектр намного лучше (остаток меньше
    вдвое — наш порог), 2D несимметрична, и слоёв больше одного.

    Возвращает словарь: verdict (текст), fwhm_1 (FWHM одного лоренциана),
    rss_ratio (остаток 4-компонентной / однокомпонентной модели), i2d_ig,
    fit (x, y, single, bilayer, components) и warnings.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x > pos_2d - half) & (x < pos_2d + half)
    xs, ys = x[keep], y[keep]
    if xs.size < 20:
        raise ValueError("Слишком мало точек вокруг 2D")
    warnings = []
    # Сдвиги для ближайшего лазера из статьи
    known = min(GRAPHENE_BILAYER_SPLIT, key=lambda lam: abs(lam - laser_nm))
    split = np.array(GRAPHENE_BILAYER_SPLIT[known])
    if abs(known - laser_nm) > 20:
        warnings.append(f"Расщепление 2D двухслойного графена в статье дано для 514.5 и 633 нм; "
                        f"для {laser_nm} нм взяты значения {known} нм.")

    edge = max(3, xs.size // 20)
    b0, a0 = np.polyfit(np.r_[xs[:edge], xs[-edge:]] - pos_2d,
                        np.r_[ys[:edge], ys[-edge:]], 1)
    h = max((ys - a0 - b0 * (xs - pos_2d)).max(), 1e-12)

    def single(xx, c, w, amp, a, b):
        return _lorentz_sum(xx, [c], w, [amp], a, b, pos_2d)

    p1, _ = curve_fit(single, xs, ys, p0=[pos_2d, 30, h, a0, b0],
                      bounds=([pos_2d - 50, 5, 0, -np.inf, -np.inf],
                              [pos_2d + 50, 150, np.inf, np.inf, np.inf]), maxfev=20000)
    rss1 = np.sum((ys - single(xs, *p1)) ** 2)

    def bilayer(xx, c, w, h1, h2, h3, h4, a, b):
        return _lorentz_sum(xx, c + split, w, [h1, h2, h3, h4], a, b, pos_2d)

    p4, _ = curve_fit(bilayer, xs, ys, p0=[pos_2d, 25, h / 3, h, h, h / 3, a0, b0],
                      bounds=([pos_2d - 50, 5, 0, 0, 0, 0, -np.inf, -np.inf],
                              [pos_2d + 50, 80, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                      maxfev=20000)
    rss4 = np.sum((ys - bilayer(xs, *p4)) ** 2)
    ratio = rss4 / max(rss1, 1e-30)

    fwhm_1 = p1[1]
    i2d_ig = p1[2] / height_g if height_g > 0 else np.nan
    if ratio > 0.5:  # 2D симметрична: одна компонента
        if fwhm_1 <= GRAPHENE_MONOLAYER_MAX_FWHM:
            verdict = "Монослой: одна узкая симметричная 2D."
            if i2d_ig < 2:
                warnings.append("Форма 2D как у монослоя, но I(2D)/I(G) < 2 (у Ferrari ≈ 4): "
                                "возможно легирование, дефекты или подложка.")
        else:
            verdict = ("Симметричная, но широкая 2D: турбостратный графит или переупакованные "
                       "чешуйки без порядка AB (например, плёнка из дисперсии). Число слоёв "
                       "по форме не определить — смотрите метрики Backes ниже.")
    elif i2d_ig < 1:
        verdict = ("Несимметричная 2D, I(2D)/I(G) < 1: больше ~5 слоёв, спектр как у графита "
                   "(Ferrari: от 5 слоёв неотличим от графита).")
    else:
        verdict = ("Несимметричная 2D из нескольких компонент: 2–5 слоёв с упаковкой AB. "
                   "Если четыре компоненты легли хорошо, а внешние слабее средних, — "
                   "вероятнее всего двухслойный.")
    warnings.append("Это оценка по форме 2D: точно число слоёв дают только эталоны с "
                    "известной толщиной (АСМ/ПЭМ), снятые на вашем приборе.")
    c4, w4, amps4 = p4[0], p4[1], p4[2:6]
    return {
        "verdict": verdict, "fwhm_1": fwhm_1, "pos_1": p1[0], "rss_ratio": ratio,
        "i2d_ig": i2d_ig,
        "fit": {"x": xs, "y": ys, "single": single(xs, *p1), "bilayer": bilayer(xs, *p4),
                "components": [lorentzian(xs, c, w4, hh) + p4[6] + p4[7] * (xs - pos_2d)
                               for c, hh in zip(c4 + split, amps4)]},
        "warnings": warnings,
    }


# ---------- Графен: расстояние между точечными дефектами ----------

# Cançado et al., Nano Lett. 11, 3190 (2011), doi:10.1021/nl201432g, уравнения 4 и 6
# (графен, облучённый ионами Ar⁺; λ в нм; I(D)/I(G) — отношение высот):
#   L_D² (нм²) = (1.8 ± 0.5)·10⁻⁹ · λ⁴ / (I_D/I_G)
#   n_D (см⁻²) = (1.8 ± 0.5)·10²² / λ⁴ · (I_D/I_G)
# Верно только на «стадии 1», при L_D ≥ 10 нм. Максимум I_D/I_G — при L_D ≈ 3 нм,
# поэтому одно и то же I_D/I_G бывает на обеих стадиях; различить их помогает
# ширина G: на стадии 1 она ≈ 14 см⁻¹ и почти не меняется, на стадии 2 — намного больше.
CANCADO_LD_COEF = (1.8e-9, 0.5e-9)
CANCADO_ND_COEF = (1.8e22, 0.5e22)
GRAPHENE_STAGE1_FWHM_G = 14.0


def graphene_defects(id_ig, laser_nm, fwhm_g=None):
    """Расстояние между дефектами L_D (нм) и их плотность n_D (см⁻²).

    Возвращает словарь: ld, ld_err, nd, nd_err, warnings. Погрешность —
    только от коэффициента статьи (~28 %); ошибка I(D)/I(G) добавляется сверху.
    """
    warnings = []
    if not id_ig > 0:
        return {"ld": np.nan, "ld_err": np.nan, "nd": np.nan, "nd_err": np.nan,
                "warnings": ["Полоса D не найдена: дефектов мало, L_D больше, чем можно оценить."]}
    k, k_err = CANCADO_LD_COEF
    ld = np.sqrt(k * laser_nm**4 / id_ig)
    ld_err = ld * 0.5 * k_err / k  # L_D ∝ √k
    kn, kn_err = CANCADO_ND_COEF
    nd = kn / laser_nm**4 * id_ig
    nd_err = nd * kn_err / kn
    if ld < 10:
        warnings.append(f"L_D = {ld:.1f} нм < 10 нм: формула верна только при L_D ≥ 10 нм "
                        "(стадия 1). Результат неверен.")
    if fwhm_g is not None and fwhm_g > 2 * GRAPHENE_STAGE1_FWHM_G:
        warnings.append(f"FWHM(G) = {fwhm_g:.0f} см⁻¹ — намного больше ~14 см⁻¹ стадии 1: "
                        "похоже на стадию 2 (сильный беспорядок), где формула не работает. "
                        "Порог 28 см⁻¹ (вдвое больше 14) — наш выбор.")
    if not 457 <= laser_nm <= 650:
        warnings.append(f"Формула проверена для лазеров 457–647 нм, а у вас {laser_nm} нм.")
    warnings.append("Формула — для точечных дефектов в монослое (облучение ионами). "
                    "Края чешуек тоже дают D, и тогда L_D занижено.")
    return {"ld": ld, "ld_err": ld_err, "nd": nd, "nd_err": nd_err, "warnings": warnings}


# ---------- Графен из жидкофазной эксфолиации: размер и толщина чешуек ----------

# Backes et al., Nanoscale (2016), doi:10.1039/C5NR08047A. Калибровка —
# на переупакованных плёнках (отфильтрованных или высохшая капля), лазер 532 нм,
# мощность < 2 мВт. Средняя длина чешуек 90–810 нм, среднее число слоёв 2.7–10.4.
#   ур. 4: <L> (мкм) = 0.094 / [(I_D/I_G)графен − (I_D/I_G)исходный графит], ±20 %
#   ур. 5: <L> (мкм) = 50 · exp(−0.21 · Γ_G), Γ_G в см⁻¹, ±20 % (верно и в жидкости)
#   ур. 6: <N> = 1.04 · M1^(−2.32), M1 = I(2D)/I(G) по высотам максимумов, ±25 %
#   ур. 8: <N> = 0.83 · exp(3.6 · M2), ±18 %, M2 — ур. 7 (см. backes_m2)
BACKES_LAYERS_RANGE = (2.7, 10.4)
BACKES_LENGTH_RANGE_UM = (0.09, 0.81)


def _band_minus_line(x, y, lo, hi):
    """Участок спектра lo–hi без линейного фона, проведённого по его краям."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x > lo) & (x < hi)
    x, y = x[keep], y[keep]
    if x.size < 20:
        raise ValueError(f"Нужен спектр в области {lo:.0f}–{hi:.0f} см⁻¹")
    edge = max(3, x.size // 20)
    b, a = np.polyfit(np.r_[x[:edge], x[-edge:]], np.r_[y[:edge], y[-edge:]], 1)
    return x, y - (a + b * x)


def backes_m1(x, y):
    """Метрика M1 (Backes 2016): отношение максимумов 2D и G, без подгонки
    (так в статье). Фон под каждой полосой — прямая по краям участка."""
    _, g = _band_minus_line(x, y, 1500, 1680)
    _, d2 = _band_minus_line(x, y, 2500, 2900)
    return d2.max() / g.max()


def backes_m2(x, y, x_graphite, y_graphite, shoulder=30.0):
    """Метрика M2 (Backes 2016, ур. 7): отношение I(ω1)/I(ω1 − 30) для
    графена, делённое на то же отношение для исходного графита.
    ω1 — положение максимума 2D графита. Интенсивности — после вычитания
    линейного фона под 2D (2500–2900 см⁻¹)."""
    def ratio(xx, yy, w1=None):
        xx, yy = _band_minus_line(xx, yy, 2500, 2900)
        if w1 is None:
            w1 = xx[np.argmax(yy)]
        return np.interp(w1, xx, yy) / np.interp(w1 - shoulder, xx, yy), w1

    r_graphite, w1 = ratio(x_graphite, y_graphite)
    r_graphene, _ = ratio(x, y, w1)
    return r_graphene / r_graphite, w1


def backes_flakes(fwhm_g=None, id_ig=None, id_ig_graphite=None, i2d_ig=None, m2=None,
                  laser_nm=532, liquid=False):
    """Средняя длина <L> (мкм) и число слоёв <N> чешуек по Backes 2016.

    Считает всё, для чего хватает данных. Возвращает словарь с ключами
    l_width, l_d (мкм), n_m1, n_m2 (каждое значение — (оценка, погрешность)
    или отсутствует) и warnings.
    """
    out, warnings = {}, []
    if fwhm_g is not None:
        L = 50 * np.exp(-0.21 * fwhm_g)
        out["l_width"] = (L, 0.2 * L)
    if id_ig is not None and id_ig_graphite is not None:
        diff = id_ig - id_ig_graphite
        if diff > 0:
            L = 0.094 / diff
            out["l_d"] = (L, 0.2 * L)
        else:
            warnings.append("I(D)/I(G) графена не больше, чем у графита: длину по D не оценить.")
    if not liquid:
        if i2d_ig is not None and i2d_ig > 0:
            N = 1.04 * i2d_ig ** -2.32
            out["n_m1"] = (N, 0.25 * N)
        if m2 is not None:
            N = 0.83 * np.exp(3.6 * m2)
            out["n_m2"] = (N, 0.18 * N)
    else:
        warnings.append("Спектр в жидкости: у Backes в жидкости работает только длина по ширине G; "
                        "толщину и длину по D считайте на высохшей капле или плёнке.")
        if "l_d" in out:
            del out["l_d"]
    for key in ("l_width", "l_d"):
        if key in out and not (BACKES_LENGTH_RANGE_UM[0] * 0.8 <= out[key][0]
                               <= BACKES_LENGTH_RANGE_UM[1] * 1.2):
            warnings.append(f"<L> = {out[key][0]:.2f} мкм вне калибровки 0.09–0.81 мкм.")
    for key in ("n_m1", "n_m2"):
        if key in out and not (BACKES_LAYERS_RANGE[0] * 0.8 <= out[key][0]
                               <= BACKES_LAYERS_RANGE[1] * 1.2):
            warnings.append(f"<N> = {out[key][0]:.1f} вне калибровки 2.7–10.4 слоя.")
    if abs(laser_nm - 532) > 10:
        warnings.append(f"Калибровка Backes — при 532 нм, у вас {laser_nm} нм: "
                        "I(D)/I(G) и форма 2D зависят от лазера.")
    warnings.append("Метрики для ансамбля чешуек жидкофазной эксфолиации (переупакованная плёнка "
                    "или капля, мощность < 2 мВт), а не для отдельной чешуйки.")
    out["warnings"] = warnings
    return out


# ---------- Порошки: сажа, угли, графит — 5 полос по Sadezky 2005 ----------

# Sadezky et al., Carbon 43, 1731 (2005), doi:10.1016/j.carbon.2005.02.018.
# Лучшая комбинация (IX): G, D1, D2, D4 — лоренцианы, D3 — гауссиан.
# Начальные положения (Table 2): G 1580, D1 1360, D2 1620, D3 1500, D4 1180 см⁻¹.
# Средние значения в статье: G 1571–1598, D1 1301–1358 (растёт с энергией лазера),
# D2 1599–1624, D3 1489–1545, D4 1127–1208 см⁻¹. Отношения полос — по площадям.
# Авторы ничего не ограничивали; мы ставим широкие рамки вокруг этих диапазонов,
# чтобы полосы не менялись местами.
SOOT_BANDS = {
    #       форма,   старт, мин,  макс,  старт FWHM
    "G":  ("lorentz", 1580, 1555, 1615, 60),
    "D1": ("lorentz", 1350, 1280, 1380, 150),
    "D2": ("lorentz", 1615, 1595, 1640, 40),
    "D3": ("gauss",   1510, 1460, 1560, 150),
    "D4": ("lorentz", 1180, 1080, 1260, 200),
}


def _soot_model(x, *p):
    from ramancore.peaks import gaussian
    total = p[-2] + p[-1] * (x - 1400)
    for i, (shape, *_rest) in enumerate(SOOT_BANDS.values()):
        c, w, h = p[3 * i: 3 * i + 3]
        total = total + (lorentzian if shape == "lorentz" else gaussian)(x, c, w, h)
    return total


def fit_soot(x, y, window=(900.0, 1900.0)):
    """Разложение спектра первого порядка сажи/угля на G, D1–D4 + линейный фон.

    Возвращает словарь: bands — таблица (pandas) с положением, FWHM, высотой
    и площадью каждой полосы и отношением площадей к G; fit — данные для
    графика; warnings.
    """
    import pandas as pd

    from ramancore.peaks import gaussian, peak_area

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= window[0]) & (x <= window[1])
    xs, ys = x[keep], y[keep]
    if xs.size < 30:
        raise ValueError(f"В диапазоне {window[0]:.0f}–{window[1]:.0f} см⁻¹ слишком мало точек")
    edge = max(3, xs.size // 20)
    b0, a0 = np.polyfit(np.r_[xs[:edge], xs[-edge:]] - 1400, np.r_[ys[:edge], ys[-edge:]], 1)
    rest = ys - (a0 + b0 * (xs - 1400))
    h = max(rest.max(), 1e-12)
    p0, lower, upper = [], [], []
    for shape, c, cmin, cmax, w in SOOT_BANDS.values():
        p0 += [c, w, np.interp(c, xs, rest).clip(0.05 * h, h) * 0.5]
        lower += [cmin, 5, 0]
        upper += [cmax, 400, np.inf]
    p0 += [a0, b0]
    lower += [-np.inf, -np.inf]
    upper += [np.inf, np.inf]
    popt, pcov = curve_fit(_soot_model, xs, ys, p0=p0, bounds=(lower, upper), maxfev=40000)
    perr = np.sqrt(np.diag(pcov))

    rows, curves, warnings = [], {}, []
    for i, (name, (shape, *_r)) in enumerate(SOOT_BANDS.items()):
        c, w, hh = popt[3 * i: 3 * i + 3]
        ce, we, he = perr[3 * i: 3 * i + 3]
        area = peak_area(w, hh, "lorentz" if shape == "lorentz" else "gauss")
        rows.append({"band": name, "shape": shape, "center": c, "center_err": ce,
                     "fwhm": w, "fwhm_err": we, "height": hh, "area": area})
        curves[name] = (lorentzian if shape == "lorentz" else gaussian)(xs, c, w, hh)
        if hh < 3 * he:
            warnings.append(f"Полоса {name} почти не видна: её параметры ненадёжны.")
    bands = pd.DataFrame(rows)
    area_g = bands.loc[bands.band == "G", "area"].iloc[0]
    bands["area_to_g"] = bands["area"] / area_g
    if bands.loc[bands.band == "D3", "area_to_g"].iloc[0] < 0.05 and \
            bands.loc[bands.band == "D1", "fwhm"].iloc[0] < 80:
        warnings.append("D3 почти нет и D1 узкая: образец ближе к графиту; у Sadezky D3 и D4 "
                        "наблюдались только в саже.")
    baseline = popt[-2] + popt[-1] * (xs - 1400)
    return {"bands": bands, "fit": {"x": xs, "y": ys, "model": _soot_model(xs, *popt),
                                    "baseline": baseline, "bands": curves},
            "warnings": warnings}


# ---------- Размер кристаллитов нанографита ----------

# Cançado et al., Appl. Phys. Lett. 88, 163106 (2006), doi:10.1063/1.2196057:
#   La (нм) = 2.4·10⁻¹⁰ · λ⁴ · (A_D/A_G)⁻¹, λ в нм, A — ПЛОЩАДИ полос.
# Калибровка: нанографит после отжига, La = 20–490 нм (рентген и СТМ),
# лазеры 457.9–647 нм.
CANCADO_LA_COEF = 2.4e-10
CANCADO_LA_RANGE = (20.0, 490.0)


def crystallite_size_la(area_d_to_g, laser_nm):
    """Размер кристаллитов La, нм, по отношению площадей D/G. Возвращает (La, warnings)."""
    warnings = []
    if not area_d_to_g > 0:
        return np.nan, ["Полоса D не найдена: кристаллиты больше, чем можно оценить."]
    la = CANCADO_LA_COEF * laser_nm**4 / area_d_to_g
    if la < CANCADO_LA_RANGE[0]:
        warnings.append(f"La = {la:.1f} нм меньше 20 нм — вне калибровки. Для сильно "
                        "разупорядоченного углерода (сажа, уголь) формула неприменима: там "
                        "I(D)/I(G) растёт с размером, а не падает (Ferrari & Robertson 2000).")
    elif la > CANCADO_LA_RANGE[1]:
        warnings.append(f"La = {la:.0f} нм больше 490 нм — вне калибровки.")
    if not 455 <= laser_nm <= 650:
        warnings.append(f"Формула проверена для видимых лазеров 457.9–647 нм, а у вас {laser_nm} нм.")
    return la, warnings
