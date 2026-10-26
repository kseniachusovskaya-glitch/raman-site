"""Тесты для ramancore/library.py."""

import numpy as np
import pandas as pd
import pytest

from ramancore.library import build_library, load_library

from conftest import SYNTHETIC


def test_build_and_load(library_files):
    meta_csv, raw, out = library_files
    n = build_library(meta_csv, raw, out)
    assert n == len(SYNTHETIC)
    lib = load_library(out)
    assert lib["spectra"].shape == (n, len(lib["grid"]))
    assert np.allclose(np.linalg.norm(lib["spectra"], axis=1), 1)  # все нормированы
    assert list(lib["metadata"]["material"]) == list(SYNTHETIC)
    assert lib["params"]["baseline_method"] == "arpls"  # параметры сохранились


def test_custom_params_are_saved(library_files):
    meta_csv, raw, out = library_files
    build_library(meta_csv, raw, out, params={"xmin": 150, "xmax": 1000})
    lib = load_library(out)
    assert lib["grid"][0] == 150 and lib["grid"][-1] == 1000
    assert lib["params"]["xmin"] == 150


@pytest.mark.parametrize("column", ["license", "source"])
def test_missing_license_or_source_rejected(library_files, column):
    # Без источника и лицензии эталон в базу не попадает
    meta_csv, raw, out = library_files
    meta = pd.read_csv(meta_csv)
    meta.loc[0, column] = ""
    meta.to_csv(meta_csv, index=False)
    with pytest.raises(ValueError, match=column):
        build_library(meta_csv, raw, out)


def test_missing_column_and_duplicate_id_rejected(library_files):
    meta_csv, raw, out = library_files
    meta = pd.read_csv(meta_csv)
    meta.drop(columns="laser_nm").to_csv(meta_csv, index=False)
    with pytest.raises(ValueError, match="laser_nm"):
        build_library(meta_csv, raw, out)
    meta.loc[1, "id"] = meta.loc[0, "id"]
    meta.to_csv(meta_csv, index=False)
    with pytest.raises(ValueError, match="id"):
        build_library(meta_csv, raw, out)


def test_several_metadata_tables(library_files):
    meta_csv, raw, out = library_files
    meta = pd.read_csv(meta_csv)
    first, second = meta_csv.parent / "a.csv", meta_csv.parent / "b.csv"
    meta.iloc[:2].to_csv(first, index=False)
    meta.iloc[2:].to_csv(second, index=False)
    assert build_library([first, second], raw, out) == len(meta)
    assert load_library(out)["spectra"].dtype == np.float32


def test_own_library_is_added(library_files):
    # Свои эталоны лежат в соседней папке library_own и подхватываются сами
    meta_csv, raw, out = library_files
    build_library(meta_csv, raw, out)
    own = pd.read_csv(meta_csv).head(2)
    own["id"] = ["own_1", "own_2"]
    own_csv = out.parent / "metadata_own.csv"
    own.to_csv(own_csv, index=False)
    build_library(own_csv, raw, out.parent / "library_own")
    lib = load_library(out)
    assert list(lib["metadata"]["id"][-2:]) == ["own_1", "own_2"]
    assert lib["spectra"].shape[0] == len(lib["metadata"])
    assert len(load_library(out, include_own=False)["metadata"]) == len(lib["metadata"]) - 2
