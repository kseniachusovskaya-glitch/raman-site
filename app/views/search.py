"""Страница: загрузка спектра → обработка → поиск по базе → выгрузка CSV.

Streamlit каждый раз
перезапускает этот файл сверху вниз, когда пользователь что-то меняет.
"""

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramancore.export import table_to_csv, to_csv
from ramancore.library import load_library
from ramancore.matching import search
from ramancore.preprocess import crop, process, remove_spikes, subtract_baseline
from ui import color, read_upload, solvent_block

ROOT = Path(__file__).parent.parent.parent  # папка проекта

st.set_page_config(page_title="RamanHub: обработка и поиск", layout="wide")


# @st.cache_resource — загрузить базу один раз и держать в памяти,
# а не читать 47 МБ при каждом клике
@st.cache_resource
def get_library():
    return load_library(ROOT / "data" / "library")


def csv_bytes(write, *args, **kwargs):
    """Пишет CSV функцией из ramancore.export во временный файл и отдаёт байты."""
    with tempfile.TemporaryDirectory() as tmp:
        path = write(Path(tmp) / "out.csv", *args, **kwargs)
        return path.read_bytes()


library = get_library()
params = library["params"]

st.title("Рамановский спектр: обработка и поиск по базе")
st.caption(f"В базе {len(library['metadata'])} эталонов. "
           "Обработка такая же, как у эталонов базы, иначе сравнение нечестное.")

# ---------- боковая панель: параметры ----------
with st.sidebar:
    st.header("Параметры")
    laser = st.selectbox("Лазер, нм", [None, 532, 633, 785], format_func=lambda v: str(v) if v else "не указан")
    xmin, xmax = st.slider("Диапазон, см⁻¹", 50, 3500, (int(params["xmin"]), int(params["xmax"])))
    method = st.selectbox("Вычитание фона", ["arpls", "asls", "poly"],
                          index=["arpls", "asls", "poly"].index(params["baseline_method"]))
    lam = st.select_slider("Жёсткость фона lam", [1e3, 1e4, 1e5, 1e6, 1e7], value=params["lam"],
                           format_func=lambda v: f"{v:g}")
    classes = sorted(library["metadata"]["class"].dropna().unique())
    chosen = st.multiselect("Искать только среди классов", classes,
                            placeholder="все классы", help="Пусто — по всей базе")
    top = st.number_input("Сколько результатов показать", 1, 50, 10)

uploaded = st.file_uploader("Файл спектра (txt или csv: два столбца, сдвиг и интенсивность)",
                            type=["txt", "csv", "dat"])
if uploaded is None:
    st.info("Загрузите файл. Разделитель (табуляция, ; , или пробел) и десятичная "
            "запятая определяются сами; строки-заголовки пропускаются.")
    st.stop()  # дальше без файла делать нечего

try:
    x, y, meta = read_upload(uploaded)
except ValueError as error:
    st.error(f"Не удалось прочитать файл: {error}")
    st.stop()
meta["laser_nm"] = laser
x, y = solvent_block(x, y, key="search")

# ---------- 1. исходный спектр и фон ----------
st.subheader("1. Исходный спектр и фон")
xc, yc = crop(x, y, xmin, xmax)
yc = remove_spikes(yc)
_, baseline = subtract_baseline(xc, yc, method=method, lam=lam)
fig = go.Figure()
fig.add_scatter(x=xc, y=yc, name="исходный (без космических лучей)",
                line=dict(width=1, color=color("sample")))
fig.add_scatter(x=xc, y=baseline, name="фон", line=dict(dash="dash", color=color("fit")))
fig.update_layout(xaxis_title="Рамановский сдвиг, см⁻¹", yaxis_title="Интенсивность",
                  height=350, margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

# ---------- 2. обработка и поиск ----------
# Для поиска спектр обрабатывается параметрами базы; если пользователь
# поменял их в панели, предупреждаем, что HQI будет ниже
user_params = dict(params, xmin=xmin, xmax=xmax, baseline_method=method, lam=lam)
grid, yp, meta_p = process(x, y, meta, **user_params)
changed = [k for k in ("xmin", "xmax", "baseline_method", "lam") if user_params[k] != params[k]]

st.subheader("2. Поиск по базе")
if changed:
    st.warning("Параметры отличаются от тех, которыми обработана база (" + ", ".join(changed) +
               "). Для поиска используются параметры базы; изменённые влияют только на графики.")
results = search(x, y, library, top=int(top), meta=meta, classes=chosen or None)
st.dataframe(results.style.format({"hqi": "{:.1f}", "hqi_deriv": "{:.1f}", "laser_nm": "{:.0f}"}),
             use_container_width=True, hide_index=True)
st.caption("HQI — сходство формы, 100 = совпадает полностью. HQI по производным "
           "меньше зависит от остатков фона. Выше 90 — хорошее совпадение.")

# Сравнение с выбранным эталоном на одном графике
if len(results):
    pick = st.selectbox("Показать эталон на графике", results["id"],
                        format_func=lambda i: f"{results.set_index('id').loc[i, 'material']} ({i})")
    _, yq, _ = process(x, y, meta, **params)  # тот же вид, что у эталонов
    row = int(np.flatnonzero(library["metadata"]["id"].to_numpy() == pick)[0])
    fig = go.Figure()
    fig.add_scatter(x=library["grid"], y=yq, name=uploaded.name,
                    line=dict(color=color("sample")))
    offset = st.slider("Сдвинуть эталон вверх (чтобы не накладывался)", 0.0, 0.5, 0.0, 0.05)
    fig.add_scatter(x=library["grid"], y=library["spectra"][row] + offset, name=pick, opacity=0.8,
                    line=dict(color=color("ref")))
    fig.update_layout(xaxis_title="Рамановский сдвиг, см⁻¹",
                      yaxis_title="Нормированная интенсивность", height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# ---------- 3. выгрузка ----------
st.subheader("3. Скачать")
col1, col2 = st.columns(2)
name = Path(uploaded.name).stem
col1.download_button("Обработанный спектр (CSV)",
                     csv_bytes(to_csv, grid, yp, meta_p, params=user_params),
                     file_name=f"{name}_processed.csv", mime="text/csv")
col2.download_button("Результаты поиска (CSV)",
                     csv_bytes(table_to_csv, results, {"filename": uploaded.name}, params=params),
                     file_name=f"{name}_search.csv", mime="text/csv")
with st.expander("История обработки"):
    for step in meta_p["history"]:
        st.write("•", step)
