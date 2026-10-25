"""Общие «фикстуры» pytest: синтетическая база эталонов для тестов.

Положения пиков ориентировочные и нужны только для тестов
(это не эталоны для публикации).
"""

import numpy as np
import pandas as pd
import pytest

# материал: [(центр, FWHM, высота), ...]
SYNTHETIC = {
    "Si": [(520.7, 4, 1000)],
    "GaAs": [(268, 4, 1000), (292, 4, 600)],
    "4H-SiC": [(204, 4, 200), (776, 4, 600), (796, 4, 1000), (964, 6, 300)],
    "GaN": [(568, 4, 1000), (734, 6, 300)],
    "graphene": [(1350, 30, 200), (1580, 15, 600), (2680, 30, 1200)],
    "MoS2": [(384, 3, 800), (406, 4, 1000)],
}


def make_spectrum(peaks, step=1.0, noise=5.0, seed=0, xmin=100, xmax=3200):
    """Сумма лоренцианов + наклонный фон + шум."""
    x = np.arange(xmin, xmax, step)
    y = 100 + 0.02 * x
    for c, w, a in peaks:
        y = y + a / (1 + ((x - c) / (w / 2)) ** 2)
    return x, y + np.random.default_rng(seed).normal(0, noise, x.size)


@pytest.fixture
def library_files(tmp_path):
    """Создаёт data/raw и metadata.csv во временной папке, возвращает пути."""
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for i, (name, peaks) in enumerate(SYNTHETIC.items()):
        x, y = make_spectrum(peaks, seed=i)
        fname = f"{name}.txt"
        np.savetxt(raw / fname, np.column_stack([x, y]), delimiter="\t")
        rows.append({"id": f"syn_{i:03d}", "file": fname, "material": name,
                     "formula": name, "class": "semiconductor", "laser_nm": 532,
                     "source": "synthetic", "license": "CC0"})
    meta_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return meta_csv, raw, tmp_path / "library"
