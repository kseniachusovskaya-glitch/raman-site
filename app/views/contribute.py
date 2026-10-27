"""Страница «Добавить спектр в базу»: форма для коллег.

Заявка уходит в закрытый репозиторий GitHub владельца базы. Адрес
репозитория и токен берутся из секретов Streamlit (раздел [submissions]),
в коде их нет.
"""

import plotly.graph_objects as go
import streamlit as st

from ramancore.submission import FIELDS, make_submission, send_to_github, validate_submission
from ui import X_TITLE, color, read_upload

st.set_page_config(page_title="RamanHub: добавить спектр", layout="wide")
st.title("Добавить спектр в базу")
st.write("Пришлите свой спектр эталонного образца — после проверки он появится в открытой "
         "базе для поиска, с вашим именем как автора. Чем подробнее описание, тем полезнее "
         "спектр: обязательные поля отмечены звёздочкой.")


def secrets():
    """(repo, token) из секретов Streamlit или None, если отправка не настроена."""
    try:
        section = st.secrets["submissions"]
        return section["repo"], section["token"]
    except (KeyError, FileNotFoundError):
        return None


target = secrets()
if target is None:
    st.warning("Отправка пока не настроена владельцем сайта: форму можно заполнить, "
               "но отправить не получится.")

uploaded = st.file_uploader("Файл спектра (txt, csv, dat: два столбца — сдвиг и интенсивность)",
                            type=["txt", "csv", "dat"], key="contrib_file")
spectrum = None
if uploaded is not None:
    try:
        spectrum = read_upload(uploaded)
    except ValueError as error:
        st.error(f"Не удалось прочитать файл: {error}")
if spectrum is not None:
    x, y, _ = spectrum
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, line=dict(width=1, color=color("sample")))
    fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=280,
                      margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(x)} точек, {x.min():.0f}–{x.max():.0f} см⁻¹. Проверьте, что ось x — "
               "рамановский сдвиг, а не длина волны.")

GROUPS = {"sample": "Образец", "measurement": "Условия съёмки", "source": "Автор и лицензия"}


def field(f):
    """Виджет для одного поля FIELDS. Ключ виджета — contrib_<key>."""
    label = f["label"] + (" *" if f.get("required") else "")
    key = f"contrib_{f['key']}"
    if f["kind"] == "choice":
        return st.selectbox(label, f["options"], index=None, placeholder="выберите",
                            key=key, help=f.get("help"))
    if f["kind"] == "number":
        return st.number_input(label, min_value=f.get("min"), max_value=f.get("max"),
                               value=None, key=key, help=f.get("help"))
    if f["kind"] == "check":
        return st.checkbox(label, key=key, help=f.get("help"))
    return st.text_input(label, key=key, help=f.get("help")).strip()


with st.form("contrib_form"):
    meta = {}
    for group, title in GROUPS.items():
        st.subheader(title)
        items = [f for f in FIELDS if f["group"] == group]
        text_like = [f for f in items if f["kind"] != "check"]
        cols = st.columns(2)
        for i, f in enumerate(text_like):
            with cols[i % 2]:
                meta[f["key"]] = field(f)
        for f in items:
            if f["kind"] == "check":
                meta[f["key"]] = field(f)
    submitted = st.form_submit_button("Отправить на проверку", type="primary",
                                      disabled=target is None)

if submitted:
    if spectrum is None:
        st.error("Загрузите файл спектра.")
        st.stop()
    errors = validate_submission(meta, spectrum[0], spectrum[1])
    if errors:
        st.error("Исправьте, пожалуйста:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.stop()
    sub_id, files = make_submission(spectrum[0], spectrum[1], meta,
                                    uploaded.name, uploaded.getvalue())
    try:
        with st.spinner("Отправляю…"):
            send_to_github(sub_id, files, *target)
    except Exception as error:  # сеть, токен, лимиты GitHub — показываем как есть
        st.error(f"Не удалось отправить: {error}. Попробуйте позже.")
        st.stop()
    st.success(f"Спасибо! Заявка {sub_id} отправлена. После проверки спектр появится в базе.")
