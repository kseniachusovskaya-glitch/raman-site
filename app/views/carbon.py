"""Страница расчётов для углеродных материалов. Пока — аморфный углерод и DLC."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramancore.carbon import (
    GRAPHENE_REF_514, fit_dlc, fit_graphene, g_dispersion, graphene_strain_doping,
    hydrogen_content,
)
from ui import fit_figure
from ui import X_TITLE, color, show_warnings, solvent_block, upload

st.set_page_config(page_title="Raman: углерод", layout="wide")
st.title("Углеродные материалы")
KINDS = {
    "dlc": "DLC и аморфный углерод (плёнки a-C, a-C:H, ta-C)",
    "powder": "Порошки: сажа, технический и активированный уголь, графит — скоро",
    "graphene": "Графен: деформация и легирование (число слоёв и дефекты — скоро)",
    "cnt": "Углеродные нанотрубки — скоро",
    "c60": "Фуллерен C60 — скоро",
    "diamond": "Алмаз и наноалмаз — скоро",
}
kind = st.selectbox("Какой углерод исследуете?", list(KINDS), format_func=KINDS.get)

if kind == "dlc":
    st.write("Подгонка D (лоренциан) + G (профиль Брейта–Вигнера–Фано) с линейным фоном "
             "по Ferrari & Robertson (2000). Загружайте **исходный** спектр, без вычитания "
             "фона: наклон фона нужен для оценки водорода.")
    c1, c2 = st.columns(2)
    with c1:
        data = upload("Спектр DLC", key="dlc")
        laser = st.selectbox("Лазер, нм", [532, 514, 633, 785], key="dlc_laser")
    with c2:
        data2 = upload("Тот же образец на другом лазере (необязательно, для дисперсии G)",
                       key="dlc2")
        laser2 = st.selectbox("Лазер второго спектра, нм", [785, 633, 532, 514], key="dlc_laser2")
    if data is None:
        st.info("Загрузите спектр — появятся положение и ширина G, I(D)/I(G) и оценка водорода.")
        st.stop()

    x, y, _ = data
    x, y = solvent_block(x, y, key="dlc")
    try:
        r = fit_dlc(x, y)
    except (ValueError, RuntimeError) as error:  # RuntimeError — подгонка не сошлась
        st.error(f"Подгонка не удалась: {error}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pos(G), см⁻¹", f"{r['pos_g']:.1f}", help="Положение максимума G")
    m2.metric("FWHM(G), см⁻¹", f"{r['fwhm_g']:.0f} ± {r['fwhm_g_err']:.0f}")
    m3.metric("I(D)/I(G)", f"{r['id_ig']:.2f} ± {r['id_ig_err']:.2f}", help="Отношение высот")
    h, h_warn = hydrogen_content(r["m_ig"], laser)
    m4.metric("H, ат. %", "—" if np.isnan(h) else f"{h:.0f}",
              help=f"m/I(G) = {r['m_ig']:.2f} мкм; Casiraghi et al. 2005")

    f = r["fit"]
    fig = go.Figure()
    fig.add_scatter(x=f["x"], y=f["y"], name="спектр", line=dict(width=1, color=color("sample")))
    fig.add_scatter(x=f["x"], y=f["baseline"], name="фон", line=dict(dash="dash", color="#7f7f7f"))
    fig.add_scatter(x=f["x"], y=f["baseline"] + f["d"], name="D", line=dict(dash="dot", color="#2ca02c"))
    fig.add_scatter(x=f["x"], y=f["baseline"] + f["g"], name="G (BWF)", line=dict(dash="dot", color="#9467bd"))
    fig.add_scatter(x=f["x"], y=f["model"], name="сумма", line=dict(width=2, color=color("fit")))
    fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=380,
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    if data2 is not None:
        try:
            r2 = fit_dlc(data2[0], data2[1])
            disp = g_dispersion(r["pos_g"], laser, r2["pos_g"], laser2)
            st.metric("Дисперсия G, см⁻¹/нм", f"{disp:.3f}",
                      help=f"Pos(G) = {r['pos_g']:.1f} при {laser} нм и "
                           f"{r2['pos_g']:.1f} при {laser2} нм")
            st.caption("≈ 0 — графит и нанокристаллический графит; чем больше, тем больше "
                       "беспорядок и доля sp³ (Ferrari & Robertson, PRB 64, 075414, 2001).")
        except (ValueError, RuntimeError) as error:
            st.error(f"Второй спектр: {error}")

    with st.expander("Как читать результат"):
        st.markdown(
            "- **Pos(G) и I(D)/I(G)** вместе показывают стадию по «трёхстадийной модели» "
            "Ferrari & Robertson (2000): от графита к нанокристаллическому графиту, затем к "
            "a-C и ta-C. Рост доли sp³ в a-C/ta-C сдвигает G и уменьшает I(D)/I(G).\n"
            "- **FWHM(G)** растёт с беспорядком связей; у плёнок одного типа — с долей sp³ и плотностью.\n"
            "- **H** — только для a-C:H, при 514–532 нм, и примерно от 20 ат. %.\n"
            "- Сравнивайте спектры, снятые на одном лазере: положение G в аморфном углероде "
            "зависит от длины волны.")
    show_warnings(r["warnings"] + h_warn)

elif kind == "graphene":
    st.write("Подгонка G, 2D и D лоренцианами. Деформация и легирование — по положениям "
             "G и 2D (Lee et al., Nat. Commun. 3, 1024, 2012).")
    data = upload("Спектр графена", key="gr")
    if data is None:
        st.info("Загрузите спектр — появятся положения G и 2D, I(2D)/I(G), I(D)/I(G), "
                "деформация и легирование.")
        st.stop()
    x, y, _ = data
    x, y = solvent_block(x, y, key="gr")
    try:
        r = fit_graphene(x, y)
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка не удалась: {error}")
        st.stop()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pos(G), см⁻¹", f"{r['pos_g']:.1f}", help=f"FWHM {r['fwhm_g']:.1f} см⁻¹")
    m2.metric("Pos(2D), см⁻¹", f"{r.get('pos_2d', float('nan')):.1f}",
              help=f"FWHM {r.get('fwhm_2d', float('nan')):.1f} см⁻¹")
    m3.metric("I(2D)/I(G)", f"{r['i2d_ig']:.2f}", help="По высотам")
    m4.metric("I(D)/I(G)", f"{r['id_ig']:.2f}", help="0 — полоса D не найдена")
    for name, label in [("g", "G"), ("2d", "2D"), ("d", "D")]:
        if name in r["tables"]:
            fig = fit_figure(x, y, r["tables"][name], r["fits"][name], title=label)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Деформация и легирование")
    if "pos_2d" not in r:
        st.warning("Нужна полоса 2D.")
    else:
        c1, c2, c3 = st.columns(3)
        g0 = c1.number_input("ωG0 эталона, см⁻¹", value=GRAPHENE_REF_514[0], step=0.1)
        d0 = c2.number_input("ω2D0 эталона, см⁻¹", value=GRAPHENE_REF_514[1], step=0.1,
                             help="По умолчанию — Lee et al. для 514.5 нм")
        kind_strain = c3.selectbox("Деформация", ["biaxial", "uniaxial"],
                                   format_func={"biaxial": "двуосная",
                                                "uniaxial": "одноосная"}.get)
        sd = graphene_strain_doping(r["pos_g"], r["pos_2d"], (g0, d0), kind_strain)
        s1, s2 = st.columns(2)
        s1.metric("Деформация, %", f"{sd['strain_percent']:+.3f}", help="> 0 — растяжение")
        s2.metric("Сдвиг G от легирования, см⁻¹", f"{sd['dg_doping']:+.1f}")
        show_warnings(sd["warnings"])
    st.caption("Число слоёв по форме 2D и плотность дефектов по I(D)/I(G) появятся после "
               "сверки с Ferrari et al. 2006 и Cançado et al. 2011.")

else:
    st.info("Этот расчёт в работе. План: порошки — разложение на 5 полос (Sadezky 2005) и "
            "размер кристаллитов La (Cançado 2006); графен — число слоёв, дефекты, деформация "
            "и легирование, размер и толщина чешуек в дисперсии (Backes 2016).")
