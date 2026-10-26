"""Тесты для заявок со спектрами от коллег."""

import io
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from ramancore.submission import (
    library_row, make_submission, send_to_github, validate_submission,
)

GOOD = {"material": "GaAs", "formula": "GaAs", "class": "semiconductor",
        "sample_form": "монокристалл", "orientation": "(100)", "instrument": "Confotec Duo",
        "laser_nm": 532, "power_mW": 2.5, "processing": "исходный (без обработки)",
        "contributor": "Иван Иванов", "organization": "ФИАН", "email": "ivan@example.org",
        "license_ok": True, "rights_ok": True}
X = np.arange(100, 800, 1.0)
Y = np.exp(-((X - 292) / 3) ** 2)
NOW = datetime(2026, 10, 26, 19, 0, 0, tzinfo=timezone.utc)


def test_valid_submission_has_no_errors():
    assert validate_submission(GOOD, X, Y) == []


def test_missing_and_bad_fields():
    bad = dict(GOOD, material="", laser_nm="зелёный", license_ok=False, email="нет")
    errors = " ".join(validate_submission(bad))
    assert "Материал" in errors and "числом" in errors
    assert "CC BY 4.0" in errors and "E-mail" in errors


def test_short_spectrum_rejected():
    assert validate_submission(GOOD, X[:20], Y[:20])


def test_make_submission():
    # Имя как у прибора: пробелы, кириллица, % — в заявке должно стать безопасным
    sub_id, files = make_submission(X, Y, GOOD, "N567-1 x100 100% 532нм back side.csv",
                                    b"1,2\n", now=NOW)
    assert sub_id == "20261026-190000-gaas-532"
    meta = json.loads(files["metadata.json"])
    assert meta["material"] == "GaAs" and meta["n_points"] == X.size
    x, y = np.loadtxt(io.BytesIO(files["spectrum.txt"]), unpack=True)
    assert np.allclose(x, X) and np.allclose(y, Y, atol=1e-4)
    assert files["original.csv"] == b"1,2\n"
    assert meta["original_file"] == "N567-1 x100 100% 532нм back side.csv"


def test_library_row_hides_email():
    _, files = make_submission(X, Y, GOOD, now=NOW)
    row = library_row(json.loads(files["metadata.json"]), "contrib/x.txt")
    assert row["license"] == "CC BY 4.0" and row["source"] == "contrib: Иван Иванов, ФИАН"
    assert "ivan@" not in json.dumps(row, ensure_ascii=False)
    assert "Форма образца: монокристалл" in row["note"]


class FakeResponse:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_send_to_github_builds_requests():
    sent = []

    def opener(request, timeout):
        sent.append(request)
        return FakeResponse()

    sub_id, files = make_submission(X, Y, GOOD, "файл 1.txt", b"1", now=NOW)
    paths = send_to_github(sub_id, files, "user/raman-submissions", "TOKEN", opener=opener)
    assert paths[0] == f"submissions/{sub_id}/spectrum.txt"
    assert sent[0].get_method() == "PUT"
    assert sent[0].full_url.endswith(paths[0])
    assert sent[0].headers["Authorization"] == "Bearer TOKEN"
    assert all(" " not in r.full_url for r in sent)
