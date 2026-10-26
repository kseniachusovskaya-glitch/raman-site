"""Предобработка спектра: обрезка, космические лучи, сглаживание, фон, нормировка.

Все функции «чистые»: не меняют входные массивы, а возвращают новые.
"""

import numpy as np
from pybaselines import Baseline
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


def smooth(y, window=5, polyorder=3):
    """Сглаживание фильтром Савицкого–Голея.

    Идея: в окне из window соседних точек (по умолчанию 5 —
    подобрано на спектре GaAs с шагом ~5 см⁻¹; в плане было 11) данные приближаются полиномом
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


def subtract_baseline(x, y, method="asls", lam=1e5, p=0.01, poly_order=3):
    """Вычитает фон (базовую линию, например люминесценцию) через pybaselines.

    По умолчанию asls с lam = 1e5 — выбрано на спектре GaAs (~500 точек,
    шаг 6.24 см⁻¹, диапазон 100–3200 см⁻¹).

    method:
      "arpls" — асимметрично перевзвешенные наименьшие квадраты
                (Baek et al., Analyst 2015, 140, 250 — сверить);
                сам решает, какие точки — фон, а какие — пики;
      "asls"  — асимметричные наименьшие квадраты (Eilers, Boelens 2005);
                p — вес точек выше фона (меньше p → фон ниже пиков);
      "poly"  — модифицированный полином степени poly_order
                (Lieber, Mahadevan-Jansen 2003); подходит для плавного фона.

    lam — «жёсткость» фона для arpls/asls: больше lam → фон глаже.
    Важно: подходящее lam зависит от числа точек. 1e5 проверено на
    спектрах из ~150 точек (шаг ~6 см⁻¹) и ~1800 точек (шаг 0.5 см⁻¹)
    для arpls; для asls на коротких спектрах лучше lam ≈ 1e3.
    asls и poly прижимают фон к нижней границе шума (остаётся смещение
    ~2σ), arpls проходит посередине шума.

    Возвращает (y_corrected, baseline).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fitter = Baseline(x_data=x)
    if method == "arpls":
        baseline, _ = fitter.arpls(y, lam=lam)
    elif method == "asls":
        baseline, _ = fitter.asls(y, lam=lam, p=p)
    elif method == "poly":
        baseline, _ = fitter.modpoly(y, poly_order=poly_order)
    else:
        raise ValueError('method должен быть "arpls", "asls" или "poly"')
    return y - baseline, baseline


def normalize(y, how="vector"):
    """Нормировка интенсивности, чтобы сравнивать спектры разной силы.

    how:
      "max"    — максимум становится 1;
      "area"   — сумма значений становится 1 (на равномерной сетке это
                 пропорционально площади под спектром);
      "vector" — длина вектора y становится 1 (sqrt(Σy²) = 1); так удобно
                 считать HQI: он становится просто квадратом скалярного
                 произведения двух спектров.

    Возвращает новый массив y.
    """
    y = np.asarray(y, dtype=float)
    if how == "max":
        norm = np.max(y)
    elif how == "area":
        norm = np.sum(y)
    elif how == "vector":
        norm = np.linalg.norm(y)  # linalg.norm — длина вектора
    else:
        raise ValueError('how должен быть "max", "area" или "vector"')
    if norm == 0:
        raise ValueError("Нельзя нормировать: спектр нулевой")
    return y / norm


def resample(x, y, grid=None, step=1.0, fill=0.0):
    """Переносит спектр на равномерную сетку (по умолчанию шаг 1 см⁻¹).

    Нужно, чтобы спектры с разных приборов можно было сравнивать точка
    в точку (например, с эталонами из базы).

    grid — готовая сетка x; если None, строится от первого до последнего
           целого см⁻¹ внутри спектра с шагом step;
    fill — значение для точек сетки вне измеренного диапазона
           (0 — «сигнала нет»; имеет смысл после вычитания фона).

    Значения между точками считаются линейной интерполяцией. Если исходный
    шаг крупнее нового (например, 6.24 → 1 см⁻¹), новых деталей не
    появляется — точки просто ложатся на прямые между измеренными.

    Возвращает (grid, y_on_grid).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if grid is None:
        # np.ceil / np.floor — округление вверх / вниз
        grid = np.arange(np.ceil(x[0]), np.floor(x[-1]) + step / 2, step)
    grid = np.asarray(grid, dtype=float)
    return grid, np.interp(grid, x, y, left=fill, right=fill)


def average_spectra(spectra):
    """Усредняет несколько спектров одного образца (повторные измерения).

    spectra — список пар (x, y). Все спектры переносятся на ось x первого
    (np.interp), затем y усредняется по точкам. Шум падает примерно в
    √N раз, а форма пиков сохраняется, если повторы сняты в одних условиях.

    Возвращает (x, y_mean).
    """
    if len(spectra) == 0:
        raise ValueError("Нет спектров для усреднения")
    x0 = np.asarray(spectra[0][0], dtype=float)
    # np.stack складывает массивы одинаковой длины в таблицу «спектр × точка»
    ys = np.stack([np.interp(x0, np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                   for x, y in spectra])
    return x0, ys.mean(axis=0)  # axis=0 — среднее по спектрам для каждой точки


def process(x, y, meta=None, xmin=100, xmax=3200, baseline_method="asls", lam=1e5,
            smooth_window=None, norm="vector", step=1.0):
    """Весь конвейер обработки по порядку:
    crop → remove_spikes → subtract_baseline → resample → (smooth) → normalize.

    Пользовательский спектр и эталоны базы должны проходить ОДИН И ТОТ ЖЕ
    конвейер с одинаковыми параметрами, иначе сравнение (HQI) нечестное.
    Сетка общая для всех: от xmin до xmax с шагом step.

    Сглаживание идёт ПОСЛЕ переноса на общую сетку, поэтому smooth_window
    здесь — в точках общей сетки (при step = 1 это см⁻¹) и одинаково для
    любых приборов. smooth_window=None (по умолчанию) — не сглаживать.

    Для подгонки пиков используйте не этот результат, а исходный спектр
    (см. ramancore/peaks.py).

    Возвращает (grid, y_processed, meta); в meta["history"] дописываются
    все шаги с параметрами.
    """
    meta = dict(meta or {})  # копия, чтобы не менять словарь вызывающего
    history = list(meta.get("history", []))

    x, y = crop(x, y, xmin, xmax)
    history.append(f"crop: {xmin}–{xmax} см⁻¹")
    y = remove_spikes(y)
    history.append("remove_spikes: threshold=6")
    y, _ = subtract_baseline(x, y, method=baseline_method, lam=lam)
    history.append(f"subtract_baseline: {baseline_method}, lam={lam:g}")
    grid = np.arange(xmin, xmax + step / 2, step)
    grid, y = resample(x, y, grid=grid)
    history.append(f"resample: шаг {step} см⁻¹")
    if smooth_window:
        y = smooth(y, window=smooth_window)
        history.append(f"smooth: Savitzky–Golay, window={smooth_window} точек сетки, polyorder=3")
    y = normalize(y, norm)
    history.append(f"normalize: {norm}")

    meta["history"] = history
    return grid, y, meta
