"""Импорт эталонов RRUFF из скачанных zip-архивов.

Запуск (архивы скачиваются с rruff.net → Download → Raman):
    uv run python scripts/import_rruff.py путь/excellent_unoriented.zip путь/excellent_oriented.zip

Что делает:
1. Берёт из архивов только обработанные спектры (Raman_Data_Processed).
2. Кладёт их в data/raw/rruff/<имя архива>/ (эта папка не попадает в git —
   файлы слишком большие, их всегда можно скачать заново).
3. Разбирает заголовки ##... и пишет data/raw/metadata_rruff.csv.
   Файлы, которые не читаются или не попадают в диапазон 100–3200 см⁻¹,
   пропускаются, их число печатается в конце.

Условия RRUFF: ссылаться на проект — Lafuente B., Downs R. T., Yang H.,
Stone N. (2015) The power of databases: the RRUFF project. In: Highlights in
Mineralogical Crystallography, T. Armbruster, R. M. Danisi (eds.),
De Gruyter, pp. 1–30 — и на внешних авторов, если они указаны.
"""

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from ramancore.io import read_rruff_header, read_spectrum

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
LICENSE = "RRUFF: cite Lafuente et al. 2015"

# Минералы RRUFF, которые являются полупроводниками, — для фильтра по классу.
# Остальные получают class = mineral.
SEMICONDUCTORS = {
    "Silicon", "Diamond", "Moissanite",            # Si, C, SiC
    "Molybdenite",                                 # MoS2
    "Sphalerite", "Wurtzite",                      # ZnS
    "Greenockite", "Hawleyite",                    # CdS
    "Galena", "Cinnabar", "Stibnite",              # PbS, HgS, Sb2S3
    "Cuprite", "Zincite", "Cassiterite",           # Cu2O, ZnO, SnO2
    "Rutile", "Anatase", "Brookite",               # TiO2
}


def clean_formula(text):
    """RRUFF пишет индексы как CaCO_3_ и заряды как Fe^2+^ — убираем разметку."""
    return text.replace("_", "").replace("^", "")


def rotation_from_name(name):
    """Угол поворота ориентированного кристалла из имени файла: «45-000__ccw» → «45° ccw»."""
    m = re.search(r"__Raman__[^_]+__(\d+)-\d+__(ccw|depolarized)?__", name)
    if not m:
        return ""
    angle, kind = m.groups()
    return f"{angle}° {kind}" if kind else f"{angle}°"


def import_zip(zip_path):
    """Распаковывает обработанные спектры из архива и возвращает строки метаданных."""
    zip_path = Path(zip_path)
    subset = zip_path.stem  # например, excellent_unoriented
    oriented = "unoriented" not in subset
    out_dir = RAW / "rruff" / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], 0
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if "Raman_Data_Processed" in n]
        for name in names:
            target = out_dir / Path(name).name
            target.write_bytes(archive.read(name))
            try:
                header = read_rruff_header(target)
                x, y, _ = read_spectrum(target)
                laser = float(re.match(r"[\d.]+", header["RAMAN WAVELENGTH"]).group())
            except (ValueError, KeyError, AttributeError):
                skipped += 1
                continue
            if len(x) < 10 or x[-1] < 100 or x[0] > 3200:
                skipped += 1
                continue

            rruff_id = header["RRUFFID"]
            if oriented:
                orientation = "; ".join(
                    p for p in (header.get("ORIENTATION", ""), rotation_from_name(name)) if p
                ) or "oriented"
            else:
                orientation = "unoriented"
            owner = header.get("OWNER", "")
            rows.append({
                "id": f"rruff_{rruff_id}_{laser:g}_{target.stem.split('__')[-1][:8]}",
                "file": str(target.relative_to(RAW)),
                "material": header["NAMES"],
                "formula": clean_formula(header.get("IDEAL CHEMISTRY", "")) or header["NAMES"],
                "class": "semiconductor" if header["NAMES"] in SEMICONDUCTORS else "mineral",
                "phase_or_polytype": "",
                "orientation": orientation,
                "laser_nm": laser,
                "power_mW": "",
                "instrument": "",
                "temperature_K": "",
                "source": f"RRUFF {rruff_id}",
                "url": header.get("URL", ""),
                "license": LICENSE,
                "note": f"sample owner: {owner}" if owner and owner != "RRUFF" else "",
            })
    print(f"{zip_path.name}: взято {len(rows)} спектров, пропущено {skipped}")
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    all_rows = []
    for path in sys.argv[1:]:  # sys.argv — аргументы командной строки
        all_rows += import_zip(path)
    table = pd.DataFrame(all_rows)
    table.to_csv(RAW / "metadata_rruff.csv", index=False)
    print(f"Всего {len(table)} эталонов RRUFF → data/raw/metadata_rruff.csv")
