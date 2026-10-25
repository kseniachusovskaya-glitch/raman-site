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
- [x] Этап 1, шаг 3: `smooth` (Савицкий–Голей) + тесты, блокнот `03_smooth.ipynb`
