"""Общие куски интерфейса для страниц сайта (без науки: только чтение и графики)."""

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramancore.io import read_spectrum
from ramancore.peaks import curves_on_grid

X_TITLE = "Рамановский сдвиг, см⁻¹"


DEFAULT_COLORS = {"sample": "#1f77b4", "ref": "#ff7f0e", "fit": "#d62728"}


def color(name):
    """Цвет линии, выбранный в боковой панели: "sample", "ref" или "fit"."""
    return st.session_state.get(f"color_{name}", DEFAULT_COLORS[name])


def read_upload(uploaded):
    """Сохраняет загруженный файл во временный и читает его read_spectrum."""
    suffix = Path(uploaded.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(uploaded.getvalue())
    try:
        x, y, meta = read_spectrum(f.name)
    finally:
        Path(f.name).unlink()  # finally — выполнится и при ошибке
    meta["filename"] = uploaded.name
    return x, y, meta


def upload(label, key):
    """Поле загрузки + чтение. Возвращает (x, y, meta) или None."""
    uploaded = st.file_uploader(label, type=["txt", "csv", "dat"], key=key)
    if uploaded is None:
        return None
    try:
        return read_upload(uploaded)
    except ValueError as error:
        st.error(f"Не удалось прочитать файл: {error}")
        return None


def fit_figure(x, y, table, fit, xlim=None, title=None):
    """Спектр, подогнанные пики и их сумма на одном графике."""
    if xlim is None:
        xlim = (fit["x"].min() - 40, fit["x"].max() + 40)
    keep = (x > xlim[0]) & (x < xlim[1])
    fig = go.Figure()
    fig.add_scatter(x=x[keep], y=y[keep], name="спектр", mode="lines",
                    line=dict(width=1, color=color("sample")))
    xs = np.linspace(fit["x"].min(), fit["x"].max(), 600)
    baseline, components, total = curves_on_grid(table, fit, xs)
    names = list(table.index) if not isinstance(table.index[0], (int, np.integer)) else None
    for i, comp in enumerate(components):
        label = names[i] if names else f"пик {i + 1}"
        fig.add_scatter(x=xs, y=baseline + comp, name=label, line=dict(dash="dot"))
    fig.add_scatter(x=xs, y=total, name="сумма", line=dict(width=2, color=color("fit")))
    fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=360,
                      margin=dict(t=30 if title else 10), title=title)
    return fig


def show_warnings(warnings):
    for w in warnings:
        st.caption("• " + w)
