"""Страница расчётов для полупроводников A3B5 и Si.

Каждая вкладка — один расчёт из ramancore.semiconductors. Спектр
загружается один раз вверху страницы и используется всеми вкладками.
"""

import pandas as pd
import streamlit as st

from ramancore.peaks import fit_peaks
from ramancore.semiconductors import (
    LOPC_PARAMS, damage_signs, gaas_biaxial_stress, gaas_orientation,
    lplus_concentration, phonon_quality, si_stress,
)
from ui import fit_figure, show_warnings, upload

st.set_page_config(page_title="Raman: полупроводники", layout="wide")
st.title("Расчёты для полупроводников")
st.caption("Подгонка идёт по исходному спектру (без нормировки и сглаживания). "
           "Под каждым результатом — оговорки, когда ему можно доверять.")

data = upload("Спектр образца (txt или csv)", key="sample")
if data is None:
    st.info("Загрузите спектр — расчёты появятся во вкладках ниже.")
    st.stop()
x, y, meta = data
material = st.selectbox("Материал", ["GaAs", "InAs", "InSb", "Si"])

tabs = st.tabs(["Ориентация", "Концентрация (L+)", "Качество кристалла",
                "Напряжения", "Лазерное повреждение"])

# ---------- ориентация (только GaAs) ----------
with tabs[0]:
    if material != "GaAs":
        st.write("Пока только для GaAs.")
    else:
        res = gaas_orientation(x, y)
        st.metric("Ориентация поверхности", res["orientation"],
                  help="По отношению площадей I(TO)/I(LO) и правилам отбора")
        st.write(f"I(TO)/I(LO) = {res['ratio']:.3f} ± {res['ratio_err']:.3f}")
        if res.get("table") is not None:
            st.plotly_chart(fit_figure(x, y, res["table"], res["fit"]), use_container_width=True)
        show_warnings(res["warnings"])

# ---------- концентрация электронов по L+ ----------
with tabs[1]:
    if material not in LOPC_PARAMS:
        st.write("Для этого материала нет параметров модели (есть: GaAs, InAs, InSb).")
    else:
        st.write("Мода L+ — связанная плазмон-фононная мода у n-типа. "
                 "Её положение даёт концентрацию электронов.")
        res = lplus_concentration(x, y, material)
        if res.get("table") is None:
            st.warning("Моду L+ найти не удалось.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("L+, см⁻¹", f"{res['omega']:.1f} ± {res['omega_err']:.1f}")
            c2.metric("n, см⁻³", f"{res['n']:.2e}", help=f"± {res['n_err']:.1e}")
            st.plotly_chart(fit_figure(x, y, res["table"], res["fit"]), use_container_width=True)
        show_warnings(res["warnings"])

# ---------- качество кристалла ----------
with tabs[2]:
    if material not in LOPC_PARAMS:
        st.write("Пока для GaAs, InAs, InSb.")
    else:
        c1, c2 = st.columns(2)
        fwhm_instr = c1.number_input("Аппаратная ширина прибора, см⁻¹", 0.0, 30.0, 7.0, 0.1,
                                     help="Confotec Duo: 6.8–7.5 см⁻¹. 0 — не учитывать")
        ref_lo = c2.number_input("LO эталонного образца, см⁻¹ (0 — нет)", 0.0, 400.0, 0.0, 0.1,
                                 help="LO нелегированного образца, снятого на том же приборе")
        res = phonon_quality(x, y, material, instrument_fwhm=fwhm_instr or None,
                             reference_lo=ref_lo or None)
        if res["table"] is None:
            st.warning("Подгонка TO и LO не сошлась.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Доля LO", f"{res['lo_fraction']:.2f} ± {res['lo_fraction_err']:.2f}",
                      help="S(LO) / (S(TO) + S(LO)); для совершенного (100) близко к 1")
            if ref_lo:
                c2.metric("Сдвиг LO, см⁻¹", f"{res['lo_shift']:+.2f}")
            cols = [c for c in ["center", "center_err", "fwhm", "fwhm_err", "intrinsic_fwhm",
                                "area"] if c in res["table"].columns]
            st.dataframe(res["table"][cols].round(2), use_container_width=True)
            st.plotly_chart(fit_figure(x, y, res["table"], res["fit"]), use_container_width=True)
        show_warnings(res["warnings"])

# ---------- напряжения ----------
with tabs[3]:
    if material not in ("Si", "GaAs"):
        st.write("Пока для Si и GaAs (001).")
    else:
        st.write("Сдвиг пика считается от **ненапряжённого эталона**. Лучше всего — "
                 "загрузить его спектр, снятый в тот же день на том же лазере: тогда "
                 "сдвиг калибровки прибора сокращается.")
        ref = upload("Спектр эталона (необязательно)", key="stress_ref")
        guess = 520.7 if material == "Si" else LOPC_PARAMS["GaAs"]["LO"]
        name = "Si" if material == "Si" else "LO"

        def center(xx, yy):
            table, fit = fit_peaks(xx, yy, [guess], shape="voigt", margin=20)
            return table, fit

        table, fit = center(x, y)
        w, w_err = table.center[0], table.center_err[0]
        if ref is not None:
            w0 = center(ref[0], ref[1])[0].center[0]
            st.write(f"Эталон: {name} = {w0:.2f} см⁻¹ (из загруженного спектра)")
        else:
            w0 = st.number_input(f"Положение {name} без напряжений, см⁻¹", value=guess, step=0.1)
            st.caption("Без эталона точность ограничена калибровкой прибора: "
                       "±0.3 см⁻¹ дают ±130 МПа для Si и ±0.1 ГПа для GaAs.")
        if material == "Si":
            sigma, err = si_stress(w, w0, w_err)
            st.metric("Напряжение, МПа", f"{sigma:.0f} ± {err:.0f}",
                      help="Одноосное, или σxx + σyy при двуосном. < 0 — сжатие")
            st.caption("De Wolf, Semicond. Sci. Technol. 11, 139 (1996): σ = −435·Δω.")
        else:
            sigma, err = gaas_biaxial_stress(w, w0, "LO", w_err)
            st.metric("Двуосное напряжение, ГПа", f"{sigma:.3f} ± {err:.3f}",
                      help="> 0 — растяжение, < 0 — сжатие")
            st.caption("Cerdeira et al., Phys. Rev. B 5, 580 (1972); ≈ −3.5 см⁻¹/ГПа для LO.")
        st.write(f"Пик образца: {w:.2f} ± {w_err:.2f} см⁻¹, сдвиг {w - w0:+.2f} см⁻¹")
        if ref is None and abs(w - w0) > 1.5:
            st.warning("Сдвиг больше 1.5 см⁻¹ без эталона — скорее всего, это калибровка "
                       "прибора, а не напряжение. Загрузите спектр эталона.")
        st.plotly_chart(fit_figure(x, y, table, fit), use_container_width=True)

# ---------- лазерное повреждение ----------
with tabs[4]:
    if material == "Si":
        st.write("Пока для GaAs, InAs, InSb.")
    else:
        st.write("Ищутся пики продуктов разложения и окисления (As, Sb, Te и их оксиды).")
        res = damage_signs(x, y, material)
        t = res["table"]
        if t.found.any():
            st.error("Найдены признаки повреждения: " + ", ".join(t[t.found]["product"]))
        else:
            st.success("Признаков повреждения не найдено.")
        st.dataframe(pd.DataFrame({
            "продукт": t["product"], "маркеры, см⁻¹": t["markers"].apply(lambda m: ", ".join(str(v) for v in m)),
            "найдено в спектре": t["matched"].apply(lambda m: ", ".join("—" if v is None else str(v) for v in m)), "найден": t["found"],
            "источник": t["source"]}), use_container_width=True, hide_index=True)
        show_warnings(res["warnings"])

st.divider()
st.caption("Температура по сдвигу фонона требует калибровочной серии спектров — "
           "эта вкладка появится следующим шагом.")
