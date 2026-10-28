"""Сайт для рамановских спектров. Запуск из папки проекта:
    uv run streamlit run app/streamlit_app.py

Этот файл только перечисляет страницы; сами страницы — в app/views/.
Здесь только интерфейс: вся наука — в ramancore.
"""

import sys
from pathlib import Path

import streamlit as st

# Ядро ramancore берём прямо из папки проекта, а не из установленной копии:
# Streamlit Cloud не переустанавливает пакет при обновлении кода, и без
# этой строки сайт видел бы старую версию ramancore
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pages = st.navigation([
    st.Page("views/search.py", title="Обработка и поиск", default=True),
    st.Page("views/batch.py", title="Пакетная обработка"),
    st.Page("views/semiconductors.py", title="Полупроводники"),
    st.Page("views/carbon.py", title="Углерод"),
    st.Page("views/polymers.py", title="Полимеры"),
    st.Page("views/contribute.py", title="Добавить спектр в базу"),
])
# Цвета графиков — в боковой панели на всех страницах. Значения
# сохраняются в st.session_state под ключами key и читаются страницами
with st.sidebar:
    st.markdown("### RamanHub")
    st.caption("Обработка рамановских спектров, расчёты для материалов и общая база эталонов")
    with st.expander("Цвета графиков"):
        st.color_picker("Ваш спектр", "#1f77b4", key="color_sample")
        st.color_picker("Эталон из базы", "#ff7f0e", key="color_ref")
        st.color_picker("Фон и подгонка", "#d62728", key="color_fit")

pages.run()
