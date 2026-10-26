"""Чтение файлов спектров.

Единый формат спектра внутри ядра:
    x    — numpy-массив, рамановский сдвиг в см⁻¹, по возрастанию;
    y    — numpy-массив, интенсивность (той же длины, что x);
    meta — словарь с описанием: имя файла, длина волны лазера, история обработки.
"""

from pathlib import Path

import numpy as np

# Разделители, которые пробуем по очереди. None означает «любые пробелы и табуляции».
_DELIMITERS = ["\t", ";", ",", None]


def _parse_line(line, delimiter):
    """Пытается превратить строку в два числа (x, y).

    Возвращает кортеж (x, y) или None, если строка не похожа на данные
    (например, это заголовок «Raman shift, Intensity»).
    """
    parts = line.split(delimiter)  # split(None) режет по любым пробелам
    parts = [p.strip() for p in parts if p.strip() != ""]
    if delimiter != ",":
        # Русский Excel пишет десятичную запятую: «520,7;100» → 520.7 и 100
        parts = [p.replace(",", ".") for p in parts]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        # float("abc") вызывает ошибку ValueError — значит, это не числа
        return None


def _detect_delimiter(lines):
    """Выбирает разделитель, при котором больше всего строк читаются как пары чисел."""
    best, best_count = None, -1
    for delimiter in _DELIMITERS:
        count = sum(1 for line in lines if _parse_line(line, delimiter) is not None)
        if count > best_count:
            best, best_count = delimiter, count
    return best


def read_spectrum(path, laser_nm=None):
    """Читает спектр из текстового файла (txt/csv) с двумя столбцами: x и y.

    - сам определяет разделитель (табуляция, точка с запятой, запятая, пробелы);
    - понимает десятичную запятую, если столбцы разделены не запятой;
    - пропускает строки-заголовки и комментарии (всё, что не читается как два числа);
    - если столбцов больше двух, берёт первые два;
    - сортирует точки по возрастанию x.

    Возвращает (x, y, meta).
    """
    path = Path(path)
    # errors="replace" — не падать на «странных» символах в заголовках приборов
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    delimiter = _detect_delimiter(lines)
    pairs = [_parse_line(line, delimiter) for line in lines]
    pairs = [p for p in pairs if p is not None]  # выбрасываем нечисловые строки

    if len(pairs) < 2:
        raise ValueError(f"В файле {path.name} не найдено двух столбцов с числами")

    data = np.array(pairs)  # массив формы (N, 2)
    x, y = data[:, 0], data[:, 1]

    # Некоторые приборы пишут спектр от больших сдвигов к меньшим — сортируем
    order = np.argsort(x)
    x, y = x[order], y[order]

    meta = {
        "filename": path.name,
        "laser_nm": laser_nm,
        "history": [f"read_spectrum: {len(x)} точек, {x[0]:.1f}–{x[-1]:.1f} см⁻¹"],
    }
    return x, y, meta


def read_rruff_header(path):
    """Читает заголовок файла RRUFF: строки вида «##КЛЮЧ=значение».

    Значение может продолжаться на следующих строках без «##» (так бывает
    в поле ORIENTATION) — они дописываются к предыдущему ключу.
    Заголовок заканчивается на первой строке с числами.

    Возвращает словарь {"NAMES": "Calcite", "RRUFFID": "R040070", ...}.
    Сами данные спектра читает read_spectrum (строки ## он пропускает).
    """
    header, last_key = {}, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("##"):
            key, _, value = line[2:].partition("=")  # partition делит строку по первому «=»
            last_key = key.strip()
            header[last_key] = value.strip()
        elif not line.strip():
            continue  # пустая строка
        elif _parse_line(line, ",") or _parse_line(line, None):
            break  # заголовок закончился, дальше идут числа
        elif last_key:
            header[last_key] += " " + line.strip()  # продолжение предыдущего значения
    return header
