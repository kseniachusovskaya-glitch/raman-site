"""Сайт для рамановских спектров. Запуск из папки проекта:
    uv run streamlit run app/streamlit_app.py

Этот файл только перечисляет страницы; сами страницы — в app/views/.
Здесь только интерфейс: вся наука — в ramancore.
"""

import streamlit as st

pages = st.navigation([
    st.Page("views/search.py", title="Обработка и поиск", default=True),
    st.Page("views/semiconductors.py", title="Полупроводники"),
])
pages.run()
