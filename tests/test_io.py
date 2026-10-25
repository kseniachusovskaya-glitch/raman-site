"""Тесты для ramancore/io.py.

tmp_path — встроенная «фикстура» pytest: временная папка, которая
создаётся для каждого теста и удаляется после.
"""

import numpy as np
import pytest

from ramancore.io import read_spectrum


def test_reads_tab_separated(tmp_path):
    f = tmp_path / "si.txt"
    f.write_text("100\t5\n200\t7\n300\t9\n")
    x, y, meta = read_spectrum(f)
    assert np.allclose(x, [100, 200, 300])
    assert np.allclose(y, [5, 7, 9])
    assert meta["filename"] == "si.txt"


def test_skips_header_and_reads_csv(tmp_path):
    f = tmp_path / "si.csv"
    f.write_text("Raman shift (cm-1),Intensity\n100.5,1.5\n200.5,2.5\n")
    x, y, _ = read_spectrum(f)
    assert np.allclose(x, [100.5, 200.5])
    assert np.allclose(y, [1.5, 2.5])


def test_semicolon_and_spaces(tmp_path):
    f1 = tmp_path / "a.csv"
    f1.write_text("100;1\n200;2\n")
    f2 = tmp_path / "b.txt"
    f2.write_text("# comment\n100   1\n200   2\n")
    for f in (f1, f2):
        x, y, _ = read_spectrum(f)
        assert np.allclose(x, [100, 200])
        assert np.allclose(y, [1, 2])


def test_sorts_descending_x(tmp_path):
    f = tmp_path / "rev.txt"
    f.write_text("300\t9\n200\t7\n100\t5\n")
    x, y, _ = read_spectrum(f)
    assert np.allclose(x, [100, 200, 300])
    assert np.allclose(y, [5, 7, 9])  # y переставлен вместе с x


def test_synthetic_si_peak_survives_reading(tmp_path):
    # Синтетический спектр: лоренциан с центром 520.7 см⁻¹
    x = np.arange(100, 1000, 0.5)
    y = 1000 / (1 + ((x - 520.7) / 2.0) ** 2)
    f = tmp_path / "synthetic.txt"
    np.savetxt(f, np.column_stack([x, y]), delimiter="\t")
    x2, y2, _ = read_spectrum(f, laser_nm=532)
    assert x2[np.argmax(y2)] == pytest.approx(520.7, abs=0.5)


def test_raises_on_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("no numbers here\n")
    with pytest.raises(ValueError):
        read_spectrum(f)


def test_read_rruff_file(tmp_path):
    from ramancore.io import read_rruff_header

    f = tmp_path / "Calcite__R040070__Raman__532.txt"
    f.write_text(
        "##NAMES=Calcite\n##RRUFFID=R040070\n##IDEAL CHEMISTRY=CaCO_3_\n"
        "##RAMAN WAVELENGTH=532\n##ORIENTATION=Laser parallel to c.\n Fiducial mark is parallel to a\n\n"
        "   100.0     ,  5.0\n   200.0     ,  7.0\n   300.0, 9.0\n##END=\n"
    )
    header = read_rruff_header(f)
    assert header["NAMES"] == "Calcite"
    assert header["RAMAN WAVELENGTH"] == "532"
    assert header["ORIENTATION"] == "Laser parallel to c. Fiducial mark is parallel to a"
    x, y, _ = read_spectrum(f)
    assert np.allclose(x, [100, 200, 300]) and np.allclose(y, [5, 7, 9])
