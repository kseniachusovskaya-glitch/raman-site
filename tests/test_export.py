"""Тесты для ramancore/export.py."""

import numpy as np
import pandas as pd
import pytest

from ramancore.export import table_to_csv, to_csv
from ramancore.io import read_spectrum
from ramancore.peaks import find_peaks, fit_peaks
from ramancore.preprocess import process

from conftest import SYNTHETIC, make_spectrum


def test_spectrum_roundtrip_is_exact(tmp_path):
    # «Без потерь»: после записи и чтения числа совпадают бит в бит
    x, y = make_spectrum(SYNTHETIC["GaAs"], step=1.21, seed=3)
    path = to_csv(tmp_path / "s.csv", x, y)
    x2, y2, _ = read_spectrum(path)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)


def test_header_has_meta_params_and_history(tmp_path):
    x, y = make_spectrum(SYNTHETIC["Si"])
    meta = {"filename": "si.txt", "laser_nm": 532, "history": ["read", "crop"]}
    grid, yp, meta_p = process(x, y, dict(meta))
    path = to_csv(tmp_path / "p.csv", grid, yp, meta_p, params={"baseline_method": "asls"})
    head = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert "# filename: si.txt" in head and "# laser_nm: 532" in head
    assert "# baseline_method: asls" in head
    assert sum(line.startswith("# history:") for line in head) == len(meta_p["history"])
    assert meta["history"] == ["read", "crop"]  # словарь пользователя не изменён


def test_extra_columns(tmp_path):
    x = np.arange(100, 110, 1.0)
    path = to_csv(tmp_path / "e.csv", x, x * 2, extra={"baseline": x * 0 + 1})
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["raman_shift_cm-1", "intensity", "baseline"]
    assert np.array_equal(df["baseline"], np.ones(10))
    with pytest.raises(ValueError):
        to_csv(tmp_path / "bad.csv", x, x, extra={"baseline": x[:3]})


def test_peak_table_roundtrip(tmp_path):
    x, y = make_spectrum(SYNTHETIC["Si"])
    table, _ = fit_peaks(x, y, find_peaks(x, y))
    path = table_to_csv(tmp_path / "peaks.csv", table, {"filename": "si.txt"},
                        params={"shape": "lorentz"})
    back = pd.read_csv(path, comment="#")
    pd.testing.assert_frame_equal(back, table.reset_index(drop=True))
