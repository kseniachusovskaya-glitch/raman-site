"""Тесты для damage_signs (признаки лазерного повреждения)."""

import pytest

from ramancore.semiconductors import damage_signs

from conftest import make_spectrum

INAS = [(218.0, 8, 1000), (243.0, 8, 300)]  # TO и LO InAs


def _found(res):
    t = res["table"]
    return list(t[t.found]["product"])


def test_clean_inas_has_no_products():
    x, y = make_spectrum(INAS, step=1.21, xmax=900)
    assert _found(damage_signs(x, y, "InAs")) == []


def test_burned_inas_shows_crystalline_as():
    x, y = make_spectrum(INAS + [(258.0, 8, 600), (198.0, 8, 250)], step=1.21, xmax=900)
    res = damage_signs(x, y, "InAs")
    assert _found(res) == ["c-As"]
    row = res["table"].set_index("product").loc["c-As"]
    assert row.all_markers
    assert any("уменьшите мощность" in w for w in res["warnings"])


def test_oxide_on_inas():
    x, y = make_spectrum(INAS + [(369.0, 6, 500), (268.0, 6, 250)], step=1.21, xmax=900)
    assert "As2O3 арсенолит" in _found(damage_signs(x, y, "InAs"))


def test_gaas_main_as_peak_hidden_by_to():
    # c-As 260 закрыт TO GaAs (268): засчитываем по второму маркеру 195
    gaas = [(268.0, 6, 300), (291.5, 6, 1000)]
    x, y = make_spectrum(gaas + [(196.0, 8, 400)], step=1.21, xmax=900)
    res = damage_signs(x, y, "GaAs")
    assert "c-As" in _found(res)
    assert any("c-As" in w and "совпадает" in w for w in res["warnings"])


def test_insb_tellurium_and_antimony():
    insb = [(180.0, 6, 1000), (191.0, 6, 600)]
    x, y = make_spectrum(insb + [(122.0, 6, 400), (150.0, 6, 300)], step=1.21, xmax=900)
    assert {"Te", "Sb"} <= set(_found(damage_signs(x, y, "InSb")))


def test_unknown_material():
    x, y = make_spectrum(INAS)
    with pytest.raises(ValueError):
        damage_signs(x, y, "Si")
