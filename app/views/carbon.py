"""Страница расчётов для углеродных материалов: DLC, порошки, графен."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramancore.carbon import (
    C60_PHASE_NAMES, DIAMOND_NU0, GRAPHENE_REF_514, backes_flakes, fit_c60, cnt_d_to_g, find_rbm, fit_cnt_g, backes_m1, backes_m2, crystallite_size_la, diamond_sp2_ratio, diamond_stress, fit_diamond_lines, fit_dlc, fit_graphene,
    fit_soot, g_dispersion, graphene_defects, graphene_layers, graphene_strain_doping,
    hydrogen_content, nanodiamond_size,
)
from ui import fit_figure
from ui import X_TITLE, color, show_warnings, solvent_block, upload

st.set_page_config(page_title="RamanHub: углерод", layout="wide")
st.title("Углеродные материалы")
KINDS = {
    "dlc": "DLC и аморфный углерод (плёнки a-C, a-C:H, ta-C)",
    "powder": "Порошки: сажа, технический и активированный уголь, нанографит",
    "graphene": "Графен: чешуйки на подложке и дисперсии",
    "cnt": "Углеродные нанотрубки",
    "c60": "Фуллерен C60",
    "diamond": "Алмаз и наноалмаз",
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

elif kind == "powder":
    st.write("Разложение спектра первого порядка на 5 полос по Sadezky et al. (Carbon 43, "
             "1731, 2005): G, D1, D2, D4 — лоренцианы, D3 — гауссиан, плюс линейный фон. "
             "Размер кристаллитов La — по отношению площадей D1/G (Cançado et al., APL 88, "
             "163106, 2006).")
    data = upload("Спектр порошка", key="soot")
    laser = st.selectbox("Лазер, нм", [532, 514, 633, 785], key="soot_laser")
    if data is None:
        st.info("Загрузите спектр — появятся положения, ширины и площади пяти полос и La.")
        st.stop()
    x, y, _ = data
    x, y = solvent_block(x, y, key="soot")
    try:
        r = fit_soot(x, y)
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка не удалась: {error}")
        st.stop()
    b = r["bands"].set_index("band")
    la, la_warn = crystallite_size_la(b.loc["D1", "area_to_g"], laser)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pos(G), см⁻¹", f"{b.loc['G', 'center']:.1f}")
    m2.metric("FWHM(D1), см⁻¹", f"{b.loc['D1', 'fwhm']:.0f}",
              help="У Sadezky: графит 42–49, сажа 101–227 см⁻¹ — чем шире, тем больше беспорядок")
    m3.metric("A(D1)/A(G)", f"{b.loc['D1', 'area_to_g']:.2f}", help="Отношение площадей")
    m4.metric("La, нм", "—" if np.isnan(la) else f"{la:.0f}" + (" ⚠" if la_warn else ""),
              help="Cançado 2006; калибровка 20–490 нм, лазеры 457.9–647 нм")

    f = r["fit"]
    fig = go.Figure()
    fig.add_scatter(x=f["x"], y=f["y"], name="спектр", line=dict(width=1, color=color("sample")))
    fig.add_scatter(x=f["x"], y=f["baseline"], name="фон", line=dict(dash="dash", color="#7f7f7f"))
    palette = {"G": "#9467bd", "D1": "#2ca02c", "D2": "#8c564b", "D3": "#e377c2", "D4": "#17becf"}
    for name, curve in f["bands"].items():
        fig.add_scatter(x=f["x"], y=f["baseline"] + curve, name=name,
                        line=dict(dash="dot", color=palette[name]))
    fig.add_scatter(x=f["x"], y=f["model"], name="сумма", line=dict(width=2, color=color("fit")))
    fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=380,
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    table = r["bands"].rename(columns={
        "band": "полоса", "shape": "форма", "center": "положение", "center_err": "±",
        "fwhm": "FWHM", "fwhm_err": "± FWHM", "height": "высота", "area": "площадь",
        "area_to_g": "площадь / G"})
    st.dataframe(table.round(2), hide_index=True)
    with st.expander("Как читать результат"):
        st.markdown(
            "- **D1** (~1350) — края графеновых слоёв; **D2** (~1620) — поверхностные слои; "
            "**D3** (~1500) — аморфный углерод; **D4** (~1200) — полиены и ионные примеси.\n"
            "- Чем **шире D1** и больше **D3/G**, тем меньше упорядоченность (Sadezky 2005).\n"
            "- Положение D1 и отношение D1/G зависят от лазера: сравнивайте образцы на одном лазере.\n"
            "- **La** имеет смысл только для нанографита (20–490 нм). У сажи и угля кристаллиты "
            "мельче, и формула неприменима.")
    show_warnings(r["warnings"] + la_warn)

elif kind == "graphene":
    st.write("Подгонка G, 2D и D лоренцианами.")
    c1, c2 = st.columns(2)
    form = c1.radio("Образец", ["flake", "dispersion"], horizontal=True,
                    format_func={"flake": "чешуйка или плёнка на подложке",
                                 "dispersion": "дисперсия (жидкофазная эксфолиация)"}.get)
    laser = c2.selectbox("Лазер, нм", [532, 514.5, 633, 785], key="gr_laser")
    data = upload("Спектр графена", key="gr")
    if data is None:
        st.info("Загрузите спектр. Для чешуйки — число слоёв, дефекты, деформация и "
                "легирование; для дисперсии — средние размер и толщина чешуек.")
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
    m4.metric("I(D)/I(G)", f"{r['id_ig']:.2f}", help="По высотам; 0 — полоса D не найдена")
    with st.expander("Подгонка G, 2D и D"):
        for name, label in [("g", "G"), ("2d", "2D"), ("d", "D")]:
            if name in r["tables"]:
                fig = fit_figure(x, y, r["tables"][name], r["fits"][name], title=label)
                st.plotly_chart(fig, use_container_width=True)

    if form == "flake":
        st.subheader("Число слоёв")
        if "pos_2d" not in r:
            st.warning("Нужна полоса 2D.")
        else:
            try:
                lay = graphene_layers(x, y, r["pos_2d"], r["height_g"], laser)
                st.success(lay["verdict"])
                st.caption(f"FWHM 2D (один лоренциан) = {lay['fwhm_1']:.1f} см⁻¹; "
                           f"остаток 4 компонент / 1 компоненты = {lay['rss_ratio']:.2f} "
                           "(меньше 0.5 — 2D несимметрична). Ferrari et al., PRL 97, 187401 (2006).")
                f = lay["fit"]
                fig = go.Figure()
                fig.add_scatter(x=f["x"], y=f["y"], name="спектр",
                                line=dict(width=1, color=color("sample")))
                fig.add_scatter(x=f["x"], y=f["single"], name="1 лоренциан",
                                line=dict(width=2, color=color("fit")))
                fig.add_scatter(x=f["x"], y=f["bilayer"], name="4 компоненты",
                                line=dict(width=2, dash="dash", color="#2ca02c"))
                fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность",
                                  height=320, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
                show_warnings(lay["warnings"])
            except (ValueError, RuntimeError) as error:
                st.error(f"Форма 2D: {error}")

        st.subheader("Дефекты")
        d = graphene_defects(r["id_ig"], laser, r["fwhm_g"])
        s1, s2 = st.columns(2)
        s1.metric("L_D, нм", "—" if np.isnan(d["ld"]) else f"{d['ld']:.1f} ± {d['ld_err']:.1f}",
                  help="Среднее расстояние между дефектами; Cançado et al., Nano Lett. 11, 3190 (2011)")
        s2.metric("n_D, см⁻²", "—" if np.isnan(d["nd"]) else f"{d['nd']:.2e}",
                  help="Плотность дефектов")
        show_warnings(d["warnings"])

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
            st.caption("Lee et al., Nat. Commun. 3, 1024 (2012).")
            show_warnings(sd["warnings"])

    else:
        st.subheader("Размер и толщина чешуек")
        st.write("Метрики Backes et al. (Nanoscale, 2016, doi:10.1039/C5NR08047A) для графена "
                 "жидкофазной эксфолиации.")
        liquid = st.radio("Где снят спектр", [False, True], horizontal=True,
                          format_func={False: "высохшая капля или плёнка",
                                       True: "прямо в жидкости"}.get)
        graphite = upload("Спектр исходного графита (необязательно: для длины по D и метрики M2)",
                          key="gr_parent")
        id_graphite, m2 = None, None
        if graphite is not None:
            try:
                id_graphite = fit_graphene(graphite[0], graphite[1])["id_ig"]
                m2, w1 = backes_m2(x, y, graphite[0], graphite[1])
                st.caption(f"Графит: I(D)/I(G) = {id_graphite:.2f}; максимум 2D при "
                           f"{w1:.0f} см⁻¹; M2 = {m2:.2f}")
            except (ValueError, RuntimeError) as error:
                st.error(f"Спектр графита: {error}")
        try:
            m1 = backes_m1(x, y)  # по максимумам, без подгонки — как в статье
        except ValueError:
            m1 = None
        bk = backes_flakes(fwhm_g=r["fwhm_g"], id_ig=r["id_ig"], id_ig_graphite=id_graphite,
                           i2d_ig=m1, m2=m2, laser_nm=laser, liquid=liquid)
        labels = [("l_width", "<L> по ширине G, мкм", "{:.2f} ± {:.2f}"),
                  ("l_d", "<L> по I(D)/I(G), мкм", "{:.2f} ± {:.2f}"),
                  ("n_m1", "<N> по I(2D)/I(G)", "{:.1f} ± {:.1f}"),
                  ("n_m2", "<N> по форме 2D (M2)", "{:.1f} ± {:.1f}")]
        cols = st.columns(4)
        for col, (key, label, fmt) in zip(cols, labels):
            col.metric(label, fmt.format(*bk[key]) if key in bk else "—")
        st.caption("<L> — средняя длина чешуек, <N> — среднее число слоёв. Длина по D и "
                   "M2 требуют спектр исходного графита.")
        show_warnings(bk["warnings"])

elif kind == "cnt":
    st.write("Диаметр трубок по радиальной дышащей моде (RBM, 100–400 см⁻¹), тип трубок по "
             "форме G (металлические — с профилем Брейта–Вигнера–Фано) и дефектность по I(D)/I(G).")
    data = upload("Спектр нанотрубок", key="cnt")
    model = st.radio("Формула для RBM", ["environment", "pristine", "jorio"], horizontal=True,
                     format_func={"environment": "трубки в окружении (Araujo 2008)",
                                  "pristine": "свободные трубки, 227/dt",
                                  "jorio": "отдельные на SiO₂, 248/dt"}.get,
                     help="Пучки, порошок, ПАВ, подложка — первый вариант")
    if data is None:
        st.info("Загрузите спектр — появятся диаметры по RBM, тип трубок и I(D)/I(G).")
        st.stop()
    x, y, _ = data
    x, y = solvent_block(x, y, key="cnt")

    st.subheader("RBM → диаметр")
    rbm = find_rbm(x, y)
    if x.min() > 150:
        st.warning(f"Спектр начинается с {x.min():.0f} см⁻¹: RBM толстых трубок срезаны.")
    if rbm.empty:
        st.info("RBM не найдены: многостенные трубки, толстые трубки (dt > ~2.5 нм) или "
                "ни одна трубка не в резонансе с лазером.")
    else:
        col = {"environment": "dt_environment", "pristine": "dt_pristine", "jorio": "dt_jorio"}[model]
        st.dataframe(rbm[["omega", col, "height"]].rename(columns={
            "omega": "ω RBM, см⁻¹", col: "диаметр, нм", "height": "высота"}).round(2),
            hide_index=True)
        st.caption("В спектре видны только трубки в резонансе с вашим лазером: другой лазер "
                   "покажет другие диаметры. Araujo et al., PRB 77, 241403 (2008); "
                   "Jorio et al., PRL 86, 1118 (2001).")

    st.subheader("G-полоса: тип трубок")
    try:
        g = fit_cnt_g(x, y)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("G+, см⁻¹", f"{g['pos_g_plus']:.1f}")
        m2.metric("G−, см⁻¹", f"{g['pos_g_minus']:.1f}")
        m3.metric("Расщепление G, см⁻¹", f"{g['split']:.1f}")
        m4.metric("Диаметр по G, нм", "—" if np.isnan(g["dt"]) else f"{g['dt']:.2f}",
                  help="ωG− = ωG+ − C/dt² (Jorio et al., PRB 65, 155412, 2002)")
        st.success({"metallic": "G− с профилем Брейта–Вигнера–Фано: в резонансе в основном "
                                "металлические трубки (Brown et al., PRB 63, 155414, 2001).",
                    "semiconducting": "G− — лоренциан: в резонансе в основном "
                                      "полупроводниковые трубки."}[g["kind"]])
        f = g["fit"]
        fig = go.Figure()
        fig.add_scatter(x=f["x"], y=f["y"], name="спектр", line=dict(width=1, color=color("sample")))
        fig.add_scatter(x=f["x"], y=f["baseline"] + f["g_plus"], name="G+",
                        line=dict(dash="dot", color="#9467bd"))
        fig.add_scatter(x=f["x"], y=f["baseline"] + f["g_minus"], name="G−",
                        line=dict(dash="dot", color="#2ca02c"))
        fig.add_scatter(x=f["x"], y=f["model"], name="сумма", line=dict(width=2, color=color("fit")))
        fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=320,
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        show_warnings(g["warnings"])
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка G: {error}")

    st.subheader("Дефекты")
    try:
        st.metric("I(D)/I(G)", f"{cnt_d_to_g(x, y):.3f}",
                  help="По максимумам; меньше — меньше дефектов и аморфного углерода")
    except ValueError as error:
        st.error(str(error))

elif kind == "c60":
    st.write("Полимеризация C60 по моде Ag(2) (~1469 см⁻¹): у димера и полимеров она "
             "смягчается. Спектр в области 1380–1500 см⁻¹ раскладывается на полосы фаз с "
             "положениями из литературы.")
    data = upload("Спектр C60", key="c60")
    monomer = st.number_input("Ag(2) мономера на вашем приборе, см⁻¹", value=1469.0, step=0.1,
                              help="Снимите свежий C60 малой мощностью и впишите положение — "
                                   "так учитывается сдвиг калибровки")
    if data is None:
        st.info("Загрузите спектр — появятся доли сигнала мономера, димера и полимеров.")
        st.stop()
    x, y, _ = data
    x, y = solvent_block(x, y, key="c60")
    try:
        r = fit_c60(x, y, monomer)
    except (ValueError, RuntimeError) as error:
        st.error(f"Подгонка не удалась: {error}")
        st.stop()
    cols = st.columns(len(r["shares"]))
    for col, (k, v) in zip(cols, r["shares"].items()):
        col.metric(C60_PHASE_NAMES[k], f"{100 * v:.0f} %")
    st.caption(f"Максимум Ag(2): {r['peak']:.1f} см⁻¹; общая ширина полос {r['fwhm']:.1f} см⁻¹. "
               "Доли — сигнала, а не массы: сечения рассеяния фаз разные. Положения фаз: "
               "Meletov et al., ФТТ 44, 601 (2002); Liu et al., AIRAPT-20 (2005).")
    f = r["fit"]
    fig = go.Figure()
    fig.add_scatter(x=f["x"], y=f["y"], name="спектр", line=dict(width=1, color=color("sample")))
    palette = {"monomer": "#2ca02c", "dimer": "#9467bd", "orthorhombic": "#8c564b",
               "tetragonal": "#e377c2", "rhombohedral": "#17becf"}
    for k, curve in f["phases"].items():
        if r["shares"][k] > 0.01:
            fig.add_scatter(x=f["x"], y=f["baseline"] + curve, name=C60_PHASE_NAMES[k],
                            line=dict(dash="dot", color=palette[k]))
    fig.add_scatter(x=f["x"], y=f["model"], name="сумма", line=dict(width=2, color=color("fit")))
    fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=340,
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    show_warnings(r["warnings"])

elif kind == "diamond":
    form = st.radio("Образец", ["film", "nano"], horizontal=True,
                    format_func={"film": "алмазная плёнка или кристалл (напряжения)",
                                 "nano": "наноалмаз (размер частиц)"}.get)
    data = upload("Спектр алмаза", key="dia")
    if data is None:
        st.info("Загрузите спектр. Для плёнки — напряжение по сдвигу линии 1332 см⁻¹, для "
                "наноалмаза — размер частиц по форме линии; для обоих — примесь sp².")
        st.stop()
    x, y, _ = data
    x, y = solvent_block(x, y, key="dia")

    if form == "film":
        st.write("Двуосное напряжение в поликристаллической плёнке по Ager & Drory "
                 "(PRB 48, 2601, 1993). Под напряжением линия расщепляется на синглет "
                 "(ниже) и дублет (выше).")
        nu0 = st.number_input("Линия ненапряжённого алмаза на вашем приборе, см⁻¹",
                              value=DIAMOND_NU0, step=0.1,
                              help="1332 — значение статьи; лучше снять объёмный алмаз тем же прибором")
        try:
            table, fit = fit_diamond_lines(x, y)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
            st.stop()
        if len(table) == 1:
            phonon = st.radio("Одна линия — считать её", ["singlet", "doublet"], horizontal=True,
                              format_func={"singlet": "синглетом", "doublet": "дублетом"}.get,
                              help="Если расщепление не разрешено, в спектре сумма обеих; "
                                   "синглет и дублет дают разные коэффициенты")
            table.loc[0, "phonon"] = phonon
        cols = st.columns(len(table) + 1)
        for col, (_, row) in zip(cols, table.iterrows()):
            sigma = diamond_stress(row.center, row.phonon, nu0)
            col.metric(f"{'синглет' if row.phonon == 'singlet' else 'дублет'}: "
                       f"{row.center:.1f} см⁻¹", f"{sigma:+.2f} ГПа",
                       help=f"FWHM {row.fwhm:.1f} см⁻¹; σ < 0 — сжатие")
        cols[-1].metric("FWHM, см⁻¹", f"{table.fwhm.min():.1f}",
                        help="Узкая линия — хорошее качество кристалла")
        st.plotly_chart(fit_figure(x, y, table, fit, title="Линия алмаза"), use_container_width=True)
        if len(table) == 1:
            show_warnings(["Одна линия: если она уширена, расщепление может быть не разрешено — "
                           "тогда напряжение оценено грубо."])
    else:
        st.write("Размер частиц по модели фононного конфайнмента Yoshikawa et al. (APL 67, 694, "
                 "1995): у мелких частиц линия алмаза сдвигается вниз, уширяется и становится "
                 "несимметричной.")
        bulk = st.number_input("Линия объёмного алмаза на вашем приборе, см⁻¹", value=1333.0,
                               step=0.1, help="В модели 1333.0; впишите своё значение, "
                                              "чтобы учесть сдвиг калибровки")
        try:
            r = nanodiamond_size(x, y, bulk)
        except (ValueError, RuntimeError) as error:
            st.error(f"Подгонка не удалась: {error}")
            st.stop()
        m1, m2, m3 = st.columns(3)
        m1.metric("Размер, нм", f"{r['size']:.1f}",
                  help=f"Погрешность подгонки ±{r['size_err']:.2f} нм; реальная больше — "
                       "она определяется точностью самой модели")
        m2.metric("Положение, см⁻¹", f"{r['pos']:.1f}", help="По модели")
        m3.metric("FWHM, см⁻¹", f"{r['fwhm']:.0f}", help="По модели")
        f = r["fit"]
        fig = go.Figure()
        fig.add_scatter(x=f["x"], y=f["y"], name="спектр", line=dict(width=1, color=color("sample")))
        fig.add_scatter(x=f["x"], y=f["baseline"], name="фон", line=dict(dash="dash", color="#7f7f7f"))
        fig.add_scatter(x=f["x"], y=f["model"], name="модель", line=dict(width=2, color=color("fit")))
        fig.update_layout(xaxis_title=X_TITLE, yaxis_title="Интенсивность", height=340,
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        show_warnings(r["warnings"])

    st.subheader("Примесь sp²")
    try:
        ratio, sp2 = diamond_sp2_ratio(x, y)
        s1, s2 = st.columns(2)
        s1.metric("A(G)/A(алмаз)", f"{ratio:.2f}", help="Отношение площадей полосы 1480–1680 "
                                                        "и линии алмаза")
        s2.metric("sp²/sp³ ≈", f"{sp2:.3f}",
                  help="С поправкой на сечение: алмаз рассеивает ~в 60 раз слабее графита "
                       "(Wada & Solin 1981, по Yoshikawa 1995)")
        show_warnings(["Оценка sp²/sp³ грубая: сечение sp² зависит от его упорядоченности и "
                       "от лазера (коэффициент 60 — для 514 нм).",
                       "У наноалмаза полоса 1600–1650 бывает от поверхностных групп (C=O, OH), "
                       "а пики 1140 и 1490 — от трансполиацетилена на границах зёрен: это не "
                       "графит."])
    except ValueError as error:
        st.info(f"sp² не оценить: {error}")

else:
    st.info("Этот расчёт в работе.")
