"""Сайт для рамановских спектров. Запуск из папки проекта:
    uv run streamlit run app/streamlit_app.py

Этот файл только перечисляет страницы; сами страницы — в app/views/.
Здесь только интерфейс: вся наука — в ramancore.
"""

import streamlit as st

pages = st.navigation([
    st.Page("views/search.py", title="Обработка и поиск", default=True),
    st.Page("views/semiconductors.py", title="Полупроводники"),
    st.Page("views/carbon.py", title="Углерод"),
])
# Цвета графиков — в боковой панели на всех страницах. Значения
# сохраняются в st.session_state под ключами key и читаются страницами
with st.sidebar:
    with st.expander("Цвета графиков"):
        st.color_picker("Ваш спектр", "#1f77b4", key="color_sample")
        st.color_picker("Эталон из базы", "#ff7f0e", key="color_ref")
        st.color_picker("Фон и подгонка", "#d62728", key="color_fit")

pages.run()
