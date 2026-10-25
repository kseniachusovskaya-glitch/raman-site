"""Предобработка спектра: обрезка, космические лучи, сглаживание, фон, нормировка.

Все функции «чистые»: не меняют входные массивы, а возвращают новые.
"""

import numpy as np
from scipy.signal import savgol_filter


def crop(x, y, xmin=100, xmax=3200):
    """Оставляет только точки с xmin <= x <= xmax (в см⁻¹).

    Возвращает (x, y) обрезанного спектра.
    """
    x, y = np.asarray(x), np.asarray(y)
    # mask — «маска»: массив True/False той же длины, что x.
    # x[mask] берёт только те элементы, где стоит True.
    mask = (x >= xmin) & (x <= xmax)
    if not mask.any():
        raise ValueError(f"В диапазоне {xmin}–{xmax} см⁻¹ нет ни одной точки спектра")
    return x[mask], y[mask]


def _modified_z(values):
    """Модифицированный z-score: насколько точка выбивается из «типичных».

    Вместо среднего и стандартного отклонения берутся медиана и MAD
    (медиана абсолютных отклонений), поэтому сами выбросы почти не влияют
    на оценку. Коэффициент 0.6745 приводит MAD к масштабу σ для нормального шума.
    Источник: Iglewicz, Hoaglin, «How to Detect and Handle Outliers», 1993.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    mad = max(mad, 1e-12)  # защита от деления на ноль на идеально гладких данных
    return 0.6745 * (values - median) / mad


def remove_spikes(y, threshold=6, max_width=3):
    """Удаляет космические лучи (узкие выбросы шириной 1–max_width точек).

    Идея (Whitaker, Hayes, Chemometr. Intell. Lab. Syst. 2018,
    DOI 10.1016/j.chemolab.2018.06.009 — сверить):
    у космического луча резкий скачок вверх и сразу резкий скачок вниз,
    поэтому ищем выбросы не в самом спектре, а в разностях соседних точек
    dy = y[i+1] - y[i], через модифицированный z-score.

    Наше уточнение, чтобы не «срезать» узкие сильные пики (например, Si):
    выброс считается лучом, только если рядом с ним разности обычные.
    У настоящего пика склоны тянутся на несколько точек, и соседние
    разности тоже большие, а у луча — нет.

    Найденные точки заменяются линейной интерполяцией по соседям.
    Возвращает новый массив y.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    dy = np.diff(y)  # длина n-1; dy[i] — переход от точки i к точке i+1
    big = np.abs(_modified_z(dy)) > threshold

    spike = np.zeros(n, dtype=bool)
    j = 1
    while j < n - 1:
        found = False
        # Пробуем ширину луча 1, 2, ..., max_width точек, начиная с точки j
        for w in range(1, max_width + 1):
            end = j + w - 1  # последняя точка луча
            if end > n - 2:
                break
            rises = big[j - 1] and dy[j - 1] > 0  # резкий скачок вверх перед лучом
            falls = big[end] and dy[end] < 0  # резкий скачок вниз после луча
            calm_before = j - 2 < 0 or not big[j - 2]
            calm_after = end + 1 > n - 2 or not big[end + 1]
            if rises and falls and calm_before and calm_after:
                spike[j : end + 1] = True
                j = end + 1
                found = True
                break
        if not found:
            j += 1

    if not spike.any():
        return y.copy()

    # np.interp заполняет точки-лучи прямой линией между хорошими соседями
    idx = np.arange(n)
    cleaned = y.copy()
    cleaned[spike] = np.interp(idx[spike], idx[~spike], y[~spike])
    return cleaned


def smooth(y, window=11, polyorder=3):
    """Сглаживание фильтром Савицкого–Голея.

    Идея: в окне из window соседних точек данные приближаются полиномом
    степени polyorder, и точка заменяется значением этого полинома.
    В отличие от простого скользящего среднего, фильтр лучше сохраняет
    положение и высоту пиков (Savitzky, Golay, Anal. Chem. 1964, 36, 1627).

    Важно: window задаётся в ТОЧКАХ, а не в см⁻¹. Если окно шире
    полуширины пика, пик становится ниже и шире. Поэтому для подгонки
    пиков (FWHM, площадь) лучше брать несглаженный спектр.

    Возвращает новый массив y.
    """
    y = np.asarray(y, dtype=float)
    # Окно должно быть нечётным и не длиннее спектра.
    # // — деление нацело: 10 // 2 = 5
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return y.copy()  # спектр слишком короткий, сглаживать нечем
    return savgol_filter(y, window, polyorder)
