# Исходные спектры эталонов

Файлы лежат здесь как скачаны или сняты, без правки. Каждый файл описывается
строкой в `metadata.csv`. Без `source` и `license` эталон в базу не попадёт.

| Поле | Пример | Обязательно |
|---|---|---|
| id | `gaas_100_001` | да |
| file | `own/GaAs_100.txt` (путь от этой папки) | да |
| material | GaAs | да |
| formula | GaAs | да |
| class | semiconductor / polymer / mineral | да |
| phase_or_polytype | (100), 4H-SiC, monolayer | для полупроводников |
| laser_nm | 532 | да |
| power_mW | 1 | желательно |
| instrument | модель спектрометра | желательно |
| temperature_K | 295 | желательно |
| source | own / RRUFF / Open Specy / DOI статьи | да |
| license | CC BY 4.0 | да |
| note | подложка, легирование, напряжения | нет |

Сборка базы: `uv run python scripts/build_library.py`
