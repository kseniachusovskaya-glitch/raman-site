"""Страница расчётов для полимеров. Пока — фазовый состав ПВДФ и ВДФ-сополимеров."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramancore.polymers import fit_pvdf, fit_pvdf_amorphous, fit_pvdf_melt, pvdf_cf2_ratio
from ui import X_TITLE, color, fit_figure, solvent_block, upload

st.set_page_config(page_title="RamanHub: полимеры", layout="wide")
st.title("Полимеры")
KINDS = {"pvdf": "ПВДФ и сополимеры ВДФ: фазы α, β, γ (пьезо- и сегнетоэлектрическая β-фаза)"}
kind = st.selectbox("Какой полимер исследуете?", list(KINDS), format_func=KINDS.get)

st.write("Фазы различаются по маятниковым колебаниям CH₂: **α ≈ 795**, **γ ≈ 811**, "
         "**β ≈ 839 см⁻¹** (Kobayashi et al. 1975; Constantino et al. 2005), а также по "
         "ножничным CF₂: β ≈ 511, α ≈ 610 см⁻¹ (Barnakov et al. 2018). Нужен диапазон "
         "хотя бы 750–900 см⁻¹, лучше 450–950.")
st.warning("**Лазер может сам менять фазу.** У ПВДФ с оксидом графена при 532 нм и мощности "
           "от ~15 мВт (объектив 10×) β переходила в α прямо во время съёмки (Barnakov et al. "
           "2018). Снимайте на малой мощности и проверьте, что спектр не меняется от "
           "повторного накопления.")

data = upload("Спектр ПВДФ", key="pvdf")
if data is None:
    st.info("Загрузите спектр — появятся доли фаз, отношение β/α и разложение на полосы.")
    st.stop()
x, y, _ = data
x, y = solvent_block(x, y, key="pvdf")
step = float(np.median(np.diff(x)))
if step > 3:
    st.warning(f"Шаг спектра {step:.1f} см⁻¹: полосы α (795) и γ (811) на таком шаге почти "
               "не разделяются, доля γ будет завышена. Для фазового анализа нужен шаг ≤ 2 см⁻¹ "
               "(спектральное разрешение ~4 см⁻¹, как в статьях).")

method = st.radio("Метод", ["derimedved", "chapron", "purushothaman"], format_func={
    "derimedved": "α, γ, β по трём полосам (Деримедведь и др. 2025)",
    "chapron": "α и β кристалла + аморфная фаза (Chapron et al. 2021)",
    "purushothaman": "α, β, γ + аморфная фаза по спектру расплава "
                     "(Purushothaman et al. 2024)"}.get)


def bands_figure(f, table, extra=None):
    """График разложения: спектр, полосы (поверх фона), сумма."""
    fig = go.Figure()
    fig.add_scatter(x=f["x"], y=f["y"], name="спектр", line=dict(width=1, color=color("sample")))
    for comp, (_, band) in zip(f["components"], table.iterrows()):
        fig.add_scatter(x=f["x"], y=f["baseline"] + comp, line=dict(dash="dot"),
                        name=f"{band['center']:.0f} {band['phase']}")
    if extra is not None:
        fig.add_scatter(x=f["x"], y=f["baseline"] + extra[1], name=extra[0],
                        line=dict(dash="dash", color="#2ca02c"))
    fig.add_scatter(x=f["x"], y=f["model"], name="сумма", line=dict(width=2, color=color("fit")))
    fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=380,
                      margin=dict(t=10))
    return fig


if method == "derimedved":
    c1, c2, c3 = st.columns(3)
    gamma = c1.checkbox("Подгонять γ (811)", value=True,
                        help="Выключите, если γ-фазы заведомо нет: иначе подгонка может "
                             "«найти» пик в шуме")
    tfe = c2.checkbox("Сополимер с ТФЭ (полоса 827)", value=False,
                      help="У ВДФ-ТФЭ слабые полосы ТФЭ 826–829 см⁻¹ на склоне β "
                           "(Деримедведь и др. 2025)")
    shape = c3.selectbox("Форма пиков", ["voigt", "lorentz", "gauss"], format_func={
        "voigt": "Псевдо-Фойгт", "lorentz": "Лоренц", "gauss": "Гаусс"}.get)
    try:
        r = fit_pvdf(x, y, gamma=gamma, tfe=tfe, shape=shape)
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка не удалась: {error}")
        st.stop()
    fr = r["fractions"]
    cols = st.columns(len(fr) + 1)
    for col, (phase, row) in zip(cols, fr.iterrows()):
        col.metric(f"{phase}-фаза, %", f"{row['по площади, %']:.0f}",
                   help=f"По высоте: {row['по высоте, %']:.0f} %")
    cols[-1].metric("I(β)/I(α)", f"{r['beta_alpha']:.2f} ± {r['beta_alpha_err']:.2f}",
                    help="Отношение площадей 839/795")
    st.plotly_chart(fit_figure(x, y, r["table"], r["fit"], xlim=(750, 910)),
                    use_container_width=True)
    st.dataframe(r["table"].drop(columns="shape").round(3), use_container_width=True)
    st.caption("Доля фазы = I фазы / (Iα + Iγ + Iβ), как у Деримедведь и др. 2025. Это "
               "предполагает одинаковое сечение рассеяния трёх полос; калибровки по ИК или "
               "рентгену нет, поэтому проценты хороши для сравнения образцов одной серии, "
               "но не как абсолютная доля. Если доли по площади и по высоте сильно "
               "расходятся, ширины полос разные и оценка грубее.")
elif method == "chapron":
    try:
        r = fit_pvdf_amorphous(x, y)
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка не удалась: {error}")
        st.stop()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("β в кристалле, %", f"{r['f_beta']:.0f}")
    m2.metric("α в кристалле, %", f"{r['f_alpha']:.0f}")
    m3.metric("Кристалл, %", f"{r['f_cryst']:.0f}", help="Доля площади лоренцианов")
    m4.metric("Аморфная фаза, %", f"{r['f_amorphous']:.0f}", help="Доля площади гауссианов")
    f = r["fit"]
    st.plotly_chart(bands_figure(f, r["table"]), use_container_width=True)
    st.dataframe(r["table"].round(3), use_container_width=True, hide_index=True)
    st.caption(f"R² = {f['r2']:.4f}. Центры полос фиксированы, как у Chapron et al. 2021: "
               "лоренцианы кристалла 795, 874 (α) и 839 (β), гауссианы аморфной фазы 797, "
               "809, 830, 881 см⁻¹; F(α) = ΣA(α)/(ΣA(α) + A(β)). Пределы ширин (кристалл "
               "3–20, аморфная фаза 15–60 см⁻¹) — наше допущение, без них лоренциан 795 и "
               "гауссиан 797 неразличимы. Метод не учитывает γ-фазу.")

else:
    source = st.radio("Аморфная фаза", ["paper", "own"], horizontal=True, format_func={
        "paper": "Соотношение из статьи (785 нм, Kaiser RXN1)",
        "own": "Свой спектр расплава"}.get)
    melt = None
    if source == "own":
        melt_data = upload("Спектр расплава ПВДФ (≈200 °C, тот же прибор и лазер)", key="pvdf_melt")
        if melt_data is None:
            st.info("Загрузите спектр расплава: его форма в 780–920 см⁻¹ станет профилем "
                    "аморфной фазы.")
            st.stop()
        melt = melt_data[:2]
    try:
        r = fit_pvdf_melt(x, y, melt=melt)
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка не удалась: {error}")
        st.stop()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Кристалличность TC, %", f"{r['tc']:.0f}",
              help="Площадь кристаллических полос (вместе с общей 881) / вся площадь")
    m2.metric("α, %", f"{r['f_alpha']:.0f}")
    m3.metric("β, %", f"{r['f_beta']:.0f}")
    m4.metric("γ, %", f"{r['f_gamma']:.0f}")
    st.caption("α, β, γ — доли в кристаллической части; умножьте на TC, чтобы получить "
               "долю от всего образца.")
    f = r["fit"]
    crystal = r["table"].iloc[:len(f["components"])]
    st.plotly_chart(bands_figure(f, crystal, extra=("аморфная фаза", f["amorphous"])),
                    use_container_width=True)
    st.dataframe(r["table"].round(3), use_container_width=True, hide_index=True)
    melt_text = ("форма вашего спектра расплава (после вычитания прямой через края окна), "
                 "подгоняется только её масштаб" if melt is not None else
                 "8 гауссианов 802, 812, 824, 836, 857, 876, 889, 904 см⁻¹ с отношением "
                 "площадей 8:4:4:2:1:2:4:1 из спектра расплава при 200 °C (785 нм)")
    st.caption(f"R² = {f['r2']:.4f}. Окно 780–920 см⁻¹, фон — прямая. Аморфная фаза — "
               f"{melt_text}. Кристалл — Pearson VII: α 799 и 877, γ 814 и ≈830, β 842, "
               "общая α+β+γ 881 см⁻¹. TC, F(α), F(β), F(γ) — ур. (10)–(13) Purushothaman "
               "et al. 2024. Наши допущения: центры кристаллических полос ±4 см⁻¹, ширины "
               "кристалла 2–20, аморфных гауссианов 6–40 см⁻¹. Соотношение из статьи снято "
               "на другом приборе; на своём лучше снять расплав. Слабые полосы (доли "
               "меньше ~10 %) делят площадь с аморфными гауссианами, их доля — грубая "
               "оценка.")

with st.expander("Проверка по ножничным CF₂ (511 β / 610 α)"):
    if x.min() > 480 or x.max() < 640:
        st.write("В спектре нет области 480–640 см⁻¹.")
    else:
        try:
            ratio, table = pvdf_cf2_ratio(x, y)
            st.metric("I(511)/I(610)", f"{ratio:.2f}",
                      help="> 1 — больше β, < 1 — больше α")
            st.dataframe(table.drop(columns="shape").round(3), use_container_width=True)
            st.caption("Полосы слабые: на шумном спектре это лишь подтверждение, какой фазы "
                       "больше, а не число.")
        except (ValueError, RuntimeError) as error:
            st.write(f"Не удалось: {error}")

st.caption("У ориентированных (вытянутых) плёнок отношение 839/795 зависит от поляризации "
           "лазера относительно направления вытяжки: 4.0, 3.5 и 2.7 для параллельной, "
           "круговой и перпендикулярной (Constantino et al. 2005). Сравнивайте образцы "
           "при одинаковой геометрии.")
