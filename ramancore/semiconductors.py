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
