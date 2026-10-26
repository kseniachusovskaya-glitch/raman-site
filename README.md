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

## Сайт

```bash
uv run streamlit run app/streamlit_app.py   # откроется в браузере, localhost:8501
```

Загрузка спектра, графики до и после обработки, поиск по базе, выгрузка CSV.

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
- [x] `subtract_baseline` (asls по умолчанию, arpls, poly через pybaselines) + тесты, блокнот `05_baseline.ipynb`
- [x] `normalize`, `resample` и весь конвейер `process` + тесты, блокнот `06_pipeline.ipynb`
- [x] `library.py` (сборка и загрузка базы, проверка источника и лицензии), `matching.py` (HQI и HQI по производным), `scripts/build_library.py`
- [x] Импорт RRUFF (`scripts/import_rruff.py`), фильтр поиска по классу, база из 9655 эталонов
- [x] `semiconductors.py`: `gaas_orientation` — ориентация GaAs по I(TO)/I(LO), блокнот `07_search_and_gaas.ipynb`
- [x] Свои эталоны (`scripts/import_own.py`, `data/library_own/` — только локально, не в git)
- [x] `export.py`: `to_csv` (спектр + фон + параметры в шапке) и `table_to_csv` (пики, поиск), блокнот `08_export.ipynb`
- [x] Концентрация электронов по моде L+ (`coupled_modes`, `concentration_from_mode`, `lplus_concentration`) для GaAs, InAs, InSb, блокнот `09_carrier_concentration.ipynb`
- [x] `thermal.py`: температура по сдвигу фонона (модель Балканского, калибровка по своей серии), блокнот `10_temperature.ipynb`
- [x] Качество кристалла: `phonon_quality` (FWHM, сдвиг LO, доля LO) и `intrinsic_fwhm` (вычет аппаратной ширины), блокнот `11_crystal_quality.ipynb`
- [x] Напряжения: `si_stress` (De Wolf, −435 МПа/см⁻¹), `volume_change` и `gaas_biaxial_stress` для GaAs (001) (Cerdeira et al. 1972, ≈ −3.5 см⁻¹/ГПа для LO), блокнот `12_stress.ipynb`
- [x] Признаки лазерного повреждения: `damage_signs` (c-As, Sb, Te и их оксиды для GaAs, InAs, InSb)
- [x] Этап 2, шаг 1: сайт `app/streamlit_app.py` — загрузка, обработка, поиск, CSV
- [x] Этап 2, шаг 2: страница «Полупроводники» — ориентация, L+, качество, напряжения, повреждение
- [x] `carbon.py`: DLC — подгонка D + G (BWF), I(D)/I(G), водород по фону (Casiraghi 2005), дисперсия G; страница «Углерод»
- [x] `subtract_reference`: вычитание спектра растворителя или подложки (на всех страницах сайта); выбор типа углерода списком
- [x] Порошки: 5 полос G, D1–D4 (Sadezky 2005), размер кристаллитов La (Cançado 2006)
- [x] Графен: число слоёв по форме 2D (Ferrari 2006), дефекты L_D и n_D (Cançado 2011), деформация и легирование (Lee 2012), размер и толщина чешуек в дисперсиях (Backes 2016)
- [x] Нанотрубки: диаметр по RBM (Araujo 2008, Jorio 2001), металлические/полупроводниковые по форме G и диаметр по расщеплению G (Jorio 2002), I(D)/I(G)
- [x] Фуллерен C60: доли мономера, димера и полимеров по моде Ag(2) (Meletov 2002, Liu 2005) с поправкой на калибровку прибора
- [x] Графен: `fit_graphene` (G, 2D, D), `graphene_strain_doping` (Lee et al. 2012)

## Пример: анализ пика

```python
from ramancore.io import read_spectrum
from ramancore.peaks import find_peaks, fit_peaks

x, y, meta = read_spectrum("data/examples/synthetic_si.txt")
table, fit = fit_peaks(x, y, find_peaks(x, y), shape="lorentz")
print(table[["center", "center_err", "fwhm", "fwhm_err", "area", "area_err"]])
```

## Пример: выгрузка в CSV

```python
from ramancore.export import to_csv, table_to_csv

to_csv("si_spectrum.csv", x, y, meta, params={"baseline_method": "asls"})
table_to_csv("si_peaks.csv", table, meta)   # читать: pd.read_csv(path, comment="#")
```

## База эталонов

`data/library/` — собранная база: 9655 спектров RRUFF (наборы excellent_unoriented
и excellent_oriented, 1952 минерала) + собственные эталоны. Поиск по всей базе
занимает ~20 мс.

Пересборка:
```bash
uv run python scripts/import_rruff.py excellent_unoriented.zip excellent_oriented.zip
uv run python scripts/build_library.py
```

Спектры RRUFF используются со ссылкой на проект:
Lafuente B., Downs R. T., Yang H., Stone N. (2015) The power of databases: the RRUFF
project. In: *Highlights in Mineralogical Crystallography*, T. Armbruster and
R. M. Danisi, eds. Berlin, Germany, W. De Gruyter, pp. 1–30.
