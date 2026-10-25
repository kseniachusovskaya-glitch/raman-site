# raman-site

Открытый веб-инструмент для обработки и идентификации рамановских спектров
с упором на полупроводники (Si, SiC, GaN, GaAs, графен, MoS₂).

Сейчас идёт **этап 1** — научное ядро `ramancore/` (Python-пакет без интерфейса).

## Установка (macOS)

```bash
brew install git python@3.12 uv
uv sync          # создаёт .venv и ставит зависимости из pyproject.toml
uv run pytest    # запускает тесты
```

## Пример

```python
from ramancore.io import read_spectrum

x, y, meta = read_spectrum("data/examples/synthetic_si.txt", laser_nm=532)
print(meta)
```

## Прогресс

- [x] Этап 1, шаг 1: `read_spectrum` в `io.py` + тесты
- [x] Блокнот `notebooks/01_read_and_plot.ipynb`: чтение и график
- [x] Этап 1, шаг 2: `crop` и `remove_spikes` в `preprocess.py` + тесты, блокнот `02_crop_and_spikes.ipynb`
- [x] Этап 1, шаг 3: `smooth` (Савицкий–Голей, окно 5 по умолчанию) + тесты, блокнот `03_smooth.ipynb`
- [x] `peaks.py`: `find_peaks` (порог по размаху и по шуму) и `fit_peaks` (Лоренц / Гаусс / псевдо-Фойгт, линейный фон, погрешности 1σ), блокнот `04_peaks.ipynb`
- [x] `subtract_baseline` (arpls / asls / poly через pybaselines) + тесты, блокнот `05_baseline.ipynb`

## Пример: анализ пика

```python
from ramancore.io import read_spectrum
from ramancore.peaks import find_peaks, fit_peaks

x, y, meta = read_spectrum("data/examples/synthetic_si.txt")
table, fit = fit_peaks(x, y, find_peaks(x, y), shape="lorentz")
print(table[["center", "center_err", "fwhm", "fwhm_err", "area", "area_err"]])
```
