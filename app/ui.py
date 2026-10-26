"""Общие куски интерфейса для страниц сайта (без науки: только чтение и графики)."""

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramancore.io import read_spectrum
from ramancore.peaks import curves_on_grid
from ramancore.preprocess import subtract_reference

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


def solvent_block(x, y, key):
    """Блок «Вычесть растворитель / подложку». Возвращает (x, y) — исправленные или прежние."""
    with st.expander("Вычесть спектр растворителя или подложки"):
        ref = upload("Спектр чистого растворителя (подложки, кюветы)", key=f"{key}_solvent")
        if ref is None:
            return x, y
        auto = st.checkbox("Подобрать масштаб по самому сильному пику растворителя",
                           value=True, key=f"{key}_auto")
        window = None
        if not auto:
            c1, c2 = st.columns(2)
            lo = c1.number_input("Окно подбора от, см⁻¹", value=float(x.min()), key=f"{key}_lo")
            hi = c2.number_input("до, см⁻¹", value=float(x.max()), key=f"{key}_hi")
            window = (lo, hi)
        try:
            x_new, y_new, scale = subtract_reference(x, y, ref[0], ref[1], window=window)
        except ValueError as error:
            st.error(str(error))
            return x, y
        st.write(f"Масштаб растворителя: {scale:.3f}")
        fig = go.Figure()
        fig.add_scatter(x=x, y=y, name="до", line=dict(width=1, color="#7f7f7f"))
        fig.add_scatter(x=x_new, y=y_new, name="после вычитания",
                        line=dict(width=1, color=color("sample")))
        fig.update_layout(xaxis_title=X_TITLE, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("В окне подбора не должно быть пиков самого образца, иначе масштаб завышен.")
        return x_new, y_new
