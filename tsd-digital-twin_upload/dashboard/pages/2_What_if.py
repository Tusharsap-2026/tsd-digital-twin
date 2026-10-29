"""Page 2 - What-if: set the disperser and see the predicted batch; compare up to three scenarios; optimise."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import GREEN, GREY, NAVY, ORANGE, RED, header, inject_css, load_core, plot_layout, tiles
from tsd_twin import whatif as wi
from tsd_twin.config import get

inject_css()
core = load_core()
cfg, twin, recipes = core["cfg"], core["twin"], core["recipes"]
header("What-if", "Change the settings and see the predicted temperature curve, energy, batch time and every limit check")

# ---------------------------------------------------------------- base settings --
with st.sidebar:
    st.markdown("#### Scenario A (base)")
    rec_id = st.selectbox("Start from recipe", list(recipes.index), index=0)
    rec = recipes.loc[rec_id]
    family = rec["family"]
    mass = st.slider("Batch mass (kg)", 1500, 4500, int(rec["batch_mass_kg"]), 50)
    solids = st.slider("Solids (% w/w)", 40.0, 72.0, float(rec["solids_pct"]), 0.5)
    pvc = st.slider("PVC (%)", 10.0, 78.0, float(rec["pvc_pct"]), 1.0)
    hegman = st.slider("Target Hegman", 4.0, 7.5, float(rec["target_hegman"]), 0.5)
    st.markdown("**Disperser**")
    rpm = st.slider("HSD speed (rpm)", 600, 1450, int(rec["dispersion_rpm"]), 5, help="18-25 m/s tip speed = 764-1061 rpm")
    anchor = st.slider("Anchor speed (rpm)", 0, 40, 35, 1)
    minutes = st.slider("Planned dispersion (min)", 15.0, 60.0, float(rec["dispersion_min"]), 0.5)
    stop_first = st.checkbox("Stop at the first passing lab check", value=False)
    st.markdown("**Utilities and season**")
    cw_in = st.slider("Chilled-water inlet (°C)", 12.0, 28.0, 22.0, 0.5)
    cw_flow = st.slider("Chilled-water flow (m³/h)", 5.0, 15.0, 10.0, 0.5)
    t_start = st.slider("Raw-material temperature (°C)", 20.0, 36.0, 30.0, 0.5)
    ambient = st.slider("Hall temperature (°C)", 18.0, 38.0, 32.0, 0.5)
    st.markdown("**Machine condition**")
    wear = st.slider("Blade wear (0 = new)", 0.0, 0.3, 0.0, 0.05)
    foul = st.select_slider("Jacket fouling (m²K/W)", options=[0.0, 0.0003, 0.0006, 0.001, 0.0015], value=0.0)

base = wi.Settings(family, mass, solids, pvc, hegman, rpm, anchor, minutes, cw_in, cw_flow, t_start, ambient, recipe_id=rec_id,
                   hegman_plateau=float(rec["hegman_plateau"]), stop_at_first_pass=stop_first, blade_wear=wear, jacket_fouling_m2k_w=foul)

# ------------------------------------------------------------ comparison scenarios --
st.markdown("##### Comparison scenarios (same recipe and season as A unless changed)")
c1, c2 = st.columns(2)
scen = {}
for col, name in [(c1, "B"), (c2, "C")]:
    with col:
        on = st.checkbox(f"Scenario {name}", value=(name == "B"), key=f"on_{name}")
        if on:
            s1, s2, s3 = st.columns(3)
            r_ = s1.number_input("HSD rpm", 600, 1450, rpm + (50 if name == "B" else -50), 5, key=f"rpm_{name}")
            m_ = s2.number_input("minutes", 15.0, 60.0, minutes + (-5.0 if name == "B" else 5.0), 0.5, key=f"min_{name}")
            f_ = s3.number_input("CW m³/h", 5.0, 15.0, cw_flow, 0.5, key=f"flow_{name}")
            s4, s5, s6 = st.columns(3)
            i_ = s4.number_input("CW inlet °C", 12.0, 28.0, cw_in, 0.5, key=f"in_{name}")
            a_ = s5.number_input("anchor rpm", 0, 40, anchor, 1, key=f"anc_{name}")
            k_ = s6.number_input("mass kg", 1500, 4500, mass, 50, key=f"mass_{name}")
            scen[name] = replace(base, hsd_rpm=float(r_), dispersion_min=float(m_), cw_flow_m3h=float(f_), cw_inlet_c=float(i_),
                                 anchor_rpm=float(a_), batch_mass_kg=float(k_))

results = {"A": wi.run_scenario(base, cfg, twin)}
for k, s in scen.items():
    results[k] = wi.run_scenario(s, cfg, twin)

# ------------------------------------------------------------------------ optimiser --
opt_col, txt_col = st.columns([1, 4])
with opt_col:
    do_opt = st.button("Optimise kWh/t for A's recipe and season", type="primary")
    margin = st.number_input("Safety margin below limit (°C)", 0.0, 5.0, 1.0, 0.5)
if do_opt:
    with st.spinner("Searching rpm x anchor x flow x time (about 10 s)..."):
        opt = wi.optimise(base, cfg, twin, wi.Bounds(safety_margin_c=float(margin)))
    st.session_state["opt"] = opt
if "opt" in st.session_state and st.session_state["opt"].get("best") is not None:
    opt = st.session_state["opt"]
    results["Optimum"] = opt["best"]
    txt_col.success(opt["explanation"])
elif "opt" in st.session_state:
    txt_col.error(st.session_state["opt"]["explanation"])

# ------------------------------------------------------------------------- results --
colours = {"A": NAVY, "B": ORANGE, "C": GREEN, "Optimum": RED}
sm = results["A"].summary
tiles([
    ("Verdict A", sm["verdict"], "", "; ".join(sm["fails"]) or "all hard checks pass", "ok" if sm["verdict"] == "PASS" else "alarm"),
    ("Energy", f"{sm['energy_total_kwh']:.1f}", "kWh", f"{sm['specific_energy_kwh_per_t']:.2f} kWh/t"),
    ("Peak product T", f"{sm['peak_temp_c']:.1f}", "°C", f"limit {sm['temp_limit_c']} - margin {sm['margin_to_limit_c']:+.1f}",
     "alarm" if sm["margin_to_limit_c"] < 0 else ("warn" if sm["margin_to_limit_c"] < 3 else "ok")),
    ("Time to target Hegman", "-" if sm["time_to_target_hegman_min"] is None else f"{sm['time_to_target_hegman_min']:.0f}", "min",
     f"dispersion {sm['dispersion_min']:.0f} min, {sm['n_extensions']} ext."),
    ("Hegman at end", f"{sm['hegman_end_of_dispersion']:.2f}", "", f"target {sm['target_hegman']}", "ok" if sm["hegman_end_of_dispersion"] >= sm["target_hegman"] else "alarm"),
    ("Batch cycle", f"{sm['batch_cycle_min']:.0f}", "min", f"{sm['throughput_t_per_h']:.2f} t/h"),
    ("HSD load", f"{sm['hsd_load_peak_pct']:.0f}", "%", f"tip speed {sm['tip_speed_ms']:.1f} m/s", "warn" if sm["hsd_load_peak_pct"] > 95 else "ok"),
])

fig = make_subplots(rows=1, cols=3, subplot_titles=("Predicted product temperature", "Predicted grind (Hegman)", "Cumulative energy"), horizontal_spacing=0.07)
for name, r in results.items():
    c = r.curve
    fig.add_trace(go.Scatter(x=c["t_min"], y=c["T_pred_c"], name=name, line=dict(color=colours[name], width=2.2)), 1, 1)
    fig.add_trace(go.Scatter(x=c["t_min"], y=c["hegman"], name=name, line=dict(color=colours[name], width=2), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=c["t_min"], y=c["energy_cum_kwh"], name=name, line=dict(color=colours[name], width=2), showlegend=False), 1, 3)
lim = sm["temp_limit_c"]
fig.add_hline(y=lim, line=dict(color=RED, dash="dot"), row=1, col=1, annotation_text=f"limit {lim} °C")
fig.add_hline(y=lim - get(cfg, f"products.{family}.temp_alarm_margin_c"), line=dict(color=ORANGE, dash="dot", width=1), row=1, col=1)
fig.add_hline(y=hegman, line=dict(color=GREY, dash="dot"), row=1, col=2, annotation_text="target")
fig.update_xaxes(title_text="minutes", row=1, col=1); fig.update_xaxes(title_text="minutes", row=1, col=2); fig.update_xaxes(title_text="minutes", row=1, col=3)
fig.update_yaxes(title_text="°C", row=1, col=1); fig.update_yaxes(title_text="Hegman", row=1, col=2); fig.update_yaxes(title_text="kWh", row=1, col=3)
st.plotly_chart(plot_layout(fig, height=380), use_container_width=True)

st.markdown("##### Side by side")
cmp = wi.compare_scenarios(results)
show = cmp.loc[["verdict", "energy_total_kwh", "specific_energy_kwh_per_t", "peak_temp_c", "margin_to_limit_c", "breach_at_min",
                "time_to_target_hegman_min", "dispersion_min", "n_extensions", "hegman_end_of_dispersion", "batch_cycle_min", "throughput_t_per_h",
                "hsd_load_peak_pct", "tip_speed_ms", "set_hsd_rpm", "set_anchor_rpm", "set_dispersion_min", "set_cw_inlet_c", "set_cw_flow_m3h",
                "set_batch_mass_kg", "fails", "warnings"]]
show.index = ["Verdict", "Energy (kWh)", "Specific energy (kWh/t)", "Peak product T (°C)", "Margin to limit (°C)", "Breach at (min)",
              "Time to target Hegman (min)", "Dispersion (min)", "Extensions", "Hegman at end", "Batch cycle (min)", "Throughput (t/h)",
              "HSD peak load (%)", "Tip speed (m/s)", "Set: HSD rpm", "Set: anchor rpm", "Set: planned min", "Set: CW inlet °C", "Set: CW m³/h",
              "Set: mass kg", "Fails", "Warnings"]
st.dataframe(show.astype(str), use_container_width=True, height=min(60 + 36 * len(show), 850))

with st.expander("Every check for scenario A"):
    st.dataframe(results["A"].checks, use_container_width=True, hide_index=True)
    for nte in results["A"].notes:
        st.caption(nte)
st.caption(f"Calibration in use: energy factor {twin.energy_factor:.3f}, grind-time factor {twin.grind_time_factor:.3f} "
           f"({twin.n_calibration_batches} simulated calibration batches). The optimiser's energy saving from higher tip speed rests on the assumed "
           f"grind-effectiveness curve - confirm on real batches before changing recipes.")
