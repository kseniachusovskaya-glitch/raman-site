"""Страница: пакетная обработка — много спектров сразу.

Загрузили серию (карта, температуры, повторы) → общий график →
одни и те же пики во всех файлах → таблица и график «параметр — номер
точки» → поиск по базе для каждого файла → среднее. Вся наука — в
ramancore.batch; здесь только интерфейс.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ramancore.batch import fit_series, peak_ratio, search_series, sort_series, wide_table
from ramancore.export import table_to_csv, to_csv
from ramancore.library import load_library
from ramancore.preprocess import average_spectra, crop
from ui import X_TITLE, color, read_upload

ROOT = Path(__file__).parent.parent.parent
MAX_FILES = 300  # больше — тяжело для бесплатного сервера Streamlit

st.set_page_config(page_title="RamanHub: пакетная обработка", layout="wide")


@st.cache_resource
def get_library():
    return load_library(ROOT / "data" / "library")


def csv_bytes(write, *args, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        return write(Path(tmp) / "out.csv", *args, **kwargs).read_bytes()


def parse_centers(text):
    """«520.7; 300» или «268, 292» → [520.7, 300.0]. Ошибка — ValueError."""
    parts = text.replace(";", " ").replace(",", " ").split()
    return [float(p) for p in parts]


st.title("Пакетная обработка")
st.caption("Много спектров за раз: карта, серия по температуре или мощности, повторы. "
           "Число в имени файла (T300K, x_12,5, point_07) считается номером точки серии.")

uploads = st.file_uploader("Файлы спектров (можно выделить сразу много)",
                           type=["txt", "csv", "dat"], accept_multiple_files=True)
if not uploads:
    st.info("Выберите несколько файлов. Формат тот же, что на странице «Обработка и поиск».")
    st.stop()
if len(uploads) > MAX_FILES:
    st.warning(f"Загружено {len(uploads)} файлов, обрабатываю первые {MAX_FILES}.")
    uploads = uploads[:MAX_FILES]

spectra, failed = [], []
for up in uploads:
    try:
        x, y, _ = read_upload(up)
        spectra.append((up.name, x, y))
    except ValueError as error:
        failed.append(f"{up.name}: {error}")
if failed:
    with st.expander(f"Не удалось прочитать {len(failed)} файл(ов)"):
        for line in failed:
            st.write("•", line)
if not spectra:
    st.stop()
spectra = sort_series(spectra)
names = [n for n, *_ in spectra]

# ---------- 1. общий график ----------
st.subheader(f"1. Все спектры ({len(spectra)})")
c1, c2, c3 = st.columns(3)
lo_all = min(float(x.min()) for _, x, _ in spectra)
hi_all = max(float(x.max()) for _, x, _ in spectra)
xlim = c1.slider("Диапазон, см⁻¹", int(lo_all), int(np.ceil(hi_all)),
                 (int(lo_all), int(np.ceil(hi_all))))
offset = c2.slider("Сдвиг между спектрами", 0.0, 1.5, 0.0, 0.1,
                   help="Каждый спектр нормирован на свой максимум в диапазоне")
max_lines = c3.number_input("Показать не больше", 1, MAX_FILES, min(50, len(spectra)))
fig = go.Figure()
for k, (name, x, y) in enumerate(spectra[:int(max_lines)]):
    xc, yc = crop(x, y, *xlim)
    if len(xc) == 0:
        continue
    yn = (yc - yc.min()) / max(np.ptp(yc), 1e-12)  # ptp — размах max − min
    fig.add_scatter(x=xc, y=yn + k * offset, name=name, mode="lines", line=dict(width=1))
fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Нормированная интенсивность",
                  height=420, margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)

tab_peaks, tab_search, tab_mean = st.tabs(["Пики по серии", "Поиск по базе", "Среднее"])

# ---------- 2. пики ----------
with tab_peaks:
    c1, c2, c3 = st.columns([2, 1, 1])
    text = c1.text_input("Примерные центры пиков, см⁻¹ (через запятую или пробел)", "520.7",
                         help="Одни и те же для всех файлов. Перекрывающиеся пики "
                              "(TO и LO, D и G) подгоняются вместе")
    shape = c2.selectbox("Форма пика", ["lorentz", "gauss", "voigt"],
                         format_func={"lorentz": "Лоренц", "gauss": "Гаусс",
                                      "voigt": "Псевдо-Фойгт"}.get)
    margin = c3.number_input("Захват по краям, см⁻¹", 5.0, 300.0, 30.0, 5.0)
    track = st.checkbox("Пики заметно смещаются по серии (каждый файл начинать с результата "
                        "предыдущего)", help="Для нагрева, карт напряжений и т. п. Файлы "
                                             "идут по возрастанию числа в имени")
    try:
        centers = parse_centers(text)
    except ValueError:
        st.error("Центры — числа через запятую или пробел, например: 268, 292")
        st.stop()
    if not centers:
        st.stop()

    long = fit_series(spectra, centers, shape=shape, margin=margin, track=track)
    wide = wide_table(long)
    n_err = int((wide["error"] != "").sum())
    n_warn = int((wide["warning"] != "").sum())
    if n_err:
        st.error(f"Не подогнано файлов: {n_err} (см. столбец error).")
    if n_warn:
        st.warning(f"В {n_warn} файл(ах) центр упёрся в границу: пик сдвинут сильнее своей "
                   "ширины. Включите слежение за сдвигом или поправьте центры.")
    st.dataframe(wide.round(4), use_container_width=True, hide_index=True)

    # График параметра по серии
    labels = wide["label"]
    use_label = labels.notna().all() and labels.nunique() == len(labels)
    xs = labels if use_label else np.arange(1, len(wide) + 1)
    x_title = "число из имени файла" if use_label else "номер файла"
    options = {f"{v}_{p}": f"{name}, пик {p}" for p in range(1, len(centers) + 1)
               for v, name in [("center", "центр"), ("fwhm", "FWHM"), ("area", "площадь")]}
    if len(centers) >= 2:
        options["ratio"] = "отношение пиков"
    c1, c2, c3 = st.columns([2, 1, 1])
    what = c1.selectbox("Что показать на графике", list(options), format_func=options.get)
    ratio = None
    if what == "ratio":
        num = c2.selectbox("Пик в числителе", range(1, len(centers) + 1), index=1)
        den = c3.selectbox("Пик в знаменателе", range(1, len(centers) + 1), index=0)
        value = st.radio("По чему считать", ["area", "amplitude"], horizontal=True,
                         format_func={"area": "площади", "amplitude": "высоты"}.get)
        ratio = peak_ratio(long, num, den, value=value).set_index("file").loc[wide["file"]]
        ys, errs = ratio["ratio"].to_numpy(), ratio["ratio_err"].to_numpy()
    else:
        ys, errs = wide[what].to_numpy(), wide[what.replace("_", "_err_", 1)].to_numpy()
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="markers+lines", text=wide["file"],
                    error_y=dict(type="data", array=errs, visible=True),
                    marker=dict(color=color("sample")), line=dict(color=color("sample")))
    fig.update_layout(xaxis_title=x_title, yaxis_title=options[what], height=360,
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Погрешности — статистические (1σ подгонки), без учёта калибровки прибора.")

    params = {"centers": ", ".join(f"{c:g}" for c in centers), "shape": shape,
              "margin": margin, "track": track, "files": len(spectra)}
    c1, c2 = st.columns(2)
    c1.download_button("Таблица по файлам (CSV)", csv_bytes(table_to_csv, wide, params=params),
                       file_name="batch_peaks.csv", mime="text/csv")
    c2.download_button("Все пики подробно (CSV)", csv_bytes(table_to_csv, long, params=params),
                       file_name="batch_peaks_long.csv", mime="text/csv")

# ---------- 3. поиск ----------
with tab_search:
    library = get_library()
    top = st.number_input("Сколько лучших совпадений на файл", 1, 10, 1)
    if st.button(f"Искать все {len(spectra)} спектров по базе"):
        st.session_state["batch_search"] = search_series(spectra, library, top=int(top))
    result = st.session_state.get("batch_search")
    if result is not None:
        if set(result["file"]) != set(names):
            st.info("Набор файлов изменился — нажмите кнопку поиска ещё раз.")
        st.dataframe(result.style.format({"hqi": "{:.1f}", "hqi_deriv": "{:.1f}",
                                          "laser_nm": "{:.0f}"}),
                     use_container_width=True, hide_index=True)
        best = result[result["rank"] == 1]
        counts = best["material"].value_counts()  # сколько файлов на каждый материал
        st.write("Лучшее совпадение по файлам: " +
                 ", ".join(f"{m} — {n}" for m, n in counts.items()))
        st.download_button("Результаты поиска (CSV)", csv_bytes(table_to_csv, result),
                           file_name="batch_search.csv", mime="text/csv")

# ---------- 4. среднее ----------
with tab_mean:
    st.caption("Для повторных измерений одного образца: шум падает примерно в √N раз. "
               "Все спектры переносятся на ось x первого файла.")
    chosen = st.multiselect("Какие файлы усреднить", names, default=names)
    if chosen:
        picked = [(x, y) for n, x, y in spectra if n in chosen]
        xm, ym = average_spectra(picked)
        fig = go.Figure()
        fig.add_scatter(x=picked[0][0], y=picked[0][1], name=f"один файл ({chosen[0]})",
                        line=dict(width=1, color="#9a9a9a"))
        fig.add_scatter(x=xm, y=ym, name=f"среднее из {len(chosen)}",
                        line=dict(width=1.5, color=color("sample")))
        fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=360,
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        meta = {"averaged_files": "; ".join(chosen), "history": [f"average_spectra: {len(chosen)}"]}
        st.download_button("Средний спектр (CSV)", csv_bytes(to_csv, xm, ym, meta),
                           file_name="average.csv", mime="text/csv")
