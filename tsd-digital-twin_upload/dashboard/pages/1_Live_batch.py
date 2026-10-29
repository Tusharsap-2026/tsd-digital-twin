"""Page 1 - Live batch: replay of a simulated batch with the twin running alongside."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import (GREEN, GREY, NAVY, ORANGE, RED, PHASE_LABEL, add_phase_bands, batch_run, default_batch, header, inject_css,
                    load_core, plot_layout, tiles)

inject_css()
core = load_core()
cfg, batches, split = core["cfg"], core["batches"], core["split"]

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.markdown("#### Live batch")
    scope = st.radio("Batches", ["Unseen test batches", "All batches"], index=0, horizontal=False)
    ids = split[split["split_random"] == "test"].index if scope.startswith("Unseen") else batches.index
    options = [f"{bid}  {batches.loc[bid, 'recipe_id']}  {batches.loc[bid, 'start_time']:%d %b %H:%M}"
               + ("  !" if batches.loc[bid, "breached"] else "") for bid in ids]
    dflt = default_batch()
    idx = list(ids).index(dflt) if dflt in list(ids) else 0
    choice = st.selectbox("Batch", options, index=idx)
    batch_id = choice.split()[0]
    speed = st.select_slider("Replay speed", options=[1, 2, 5, 10, 30], value=10, help="samples (10 s each) per frame")
    playing = st.checkbox("Play", value=False)
    st.caption("'!' marks batches whose product temperature crossed the limit")

run = batch_run(batch_id)
meta = batches.loc[batch_id]
n = len(run)
if "t_idx" not in st.session_state or st.session_state.get("t_batch") != batch_id:
    hsd_rows = run.index[run["step"] == "high_speed_dispersion"]
    st.session_state.t_idx = int(hsd_rows[len(hsd_rows) * 2 // 3]) if len(hsd_rows) else min(60, n - 1)   # start two-thirds into dispersion
    st.session_state.t_batch = batch_id

header("Live batch", f"{batch_id} - {meta['recipe_name']} ({meta['product_family'].replace('_', '-')}), "
                     f"{meta['batch_mass_kg']:.0f} kg, target Hegman {meta['target_hegman']}, started {meta['start_time']:%d %b %Y %H:%M}")

t_idx = st.slider("Time in batch (min)", 0.0, float(run["t_min"].iloc[-1]), float(run["t_min"].iloc[st.session_state.t_idx]), 1 / 6,
                  format="%.1f", key="time_slider")
i = int(np.clip(round(t_idx * 6), 0, n - 1))
st.session_state.t_idx = i

tiles_ph = st.empty()
alert_ph = st.empty()
chart_ph = st.empty()
foot_ph = st.empty()


def render(i: int) -> None:
    now = run.iloc[i]
    limit, pre = now["limit_c"], now["prealarm_c"]
    T_state = "alarm" if now["T_meas"] > limit else ("warn" if now["T_meas"] > pre else "ok")
    nx = now["T_next5"]
    nx_state = "alarm" if (not pd.isna(nx) and nx > limit) else ("warn" if (not pd.isna(nx) and nx > pre) else "ok")
    eta = now["breach_eta_min"]
    eta_state = "alarm" if (not pd.isna(eta) and eta <= 5) else ("warn" if not pd.isna(eta) else "ok")
    lab = run["hegman_lab"].iloc[: i + 1].dropna()
    with tiles_ph.container():
        tiles([
            ("Phase", {"charge_liquids": "Charge", "pigment_addition": "Pigment", "high_speed_dispersion": "Dispersion", "letdown": "Letdown"}[now["step"]],
             "", f"t = {now['t_min']:.1f} min"),
            ("Product T", f"{now['T_meas']:.1f}", "°C", f"limit {limit:.0f} / pre-alarm {pre:.0f}", T_state),
            ("Twin now-cast", f"{now['T_nowcast']:.1f}", "°C", f"gap {now['T_meas'] - now['T_nowcast']:+.1f} °C"),
            ("T in 5 min", "-" if pd.isna(nx) else f"{nx:.1f}", "°C", "5-min-ahead model", nx_state),
            ("Breach ETA", "-" if pd.isna(eta) else f"{eta:.0f}", "min", "projected crossing of the limit" if not pd.isna(eta) else "no crossing projected", eta_state),
            ("HSD power", f"{now['hsd_kw']:.1f}", "kW", f"{now['hsd_rpm']:.0f} rpm (set {now['hsd_rpm_setpoint']:.0f}); physics {now['hsd_kw_phys']:.1f} kW"),
            ("Energy", f"{now['energy_cum_kwh']:.1f}", "kWh", f"{now['specific_energy_cum_kwh_t']:.2f} kWh/t so far"),
            ("Grind (lab)", "-" if lab.empty else f"{lab.iloc[-1]:.1f}", "Hegman", f"target {meta['target_hegman']}; twin {now['hegman_phys']:.1f}" if not pd.isna(now["hegman_phys"]) else f"target {meta['target_hegman']}"),
        ])
    code = now["alert_code"]
    cls = "alert-alarm" if code == "TEMP_LIMIT" else ("alert-none" if code == "NONE" else "alert-warn")
    text = {"NONE": "No active alert", "TEMP_LIMIT": "PRODUCT TEMPERATURE ABOVE LIMIT", "TEMP_PREALARM": "Temperature pre-alarm: limit expected within reach",
            "LOW_CW_FLOW": "Low chilled-water flow", "OVERCHARGE": "Batch overcharged", "JACKET_FOULING": "Jacket fouling suspected",
            "BLADE_WEAR": "Blade wear suspected"}[code]
    cause = now["likely_cause"] if code.startswith("TEMP") else ""
    extra = []
    for k in ["LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"]:
        if now[f"rule_{k}"] and k != code:
            extra.append(k.replace("_", " ").lower())
    alert_ph.markdown(f'<div class="alert-strip {cls}">{text}' + (f" &mdash; likely cause: {cause}" if cause else "")
                      + (f" &nbsp;|&nbsp; also: {', '.join(extra)}" if extra else "") + "</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Product temperature - measured vs twin", "HSD power and speed", "Energy counters"),
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]])
    past, fut = run.iloc[: i + 1], run.iloc[i:]
    fig.add_trace(go.Scatter(x=run["t_min"], y=run["T_forecast"], name="pre-batch forecast", line=dict(color=GREY, width=1.2, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=past["t_min"], y=past["T_nowcast"], name="twin now-cast", line=dict(color=ORANGE, width=1.5, dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=past["t_min"], y=past["T_meas"], name="measured", line=dict(color=NAVY, width=2.2)), 1, 1)
    if not pd.isna(nx):
        fig.add_trace(go.Scatter(x=[now["t_min"], now["t_min"] + 5], y=[now["T_meas"], nx], name="5-min-ahead", mode="lines+markers",
                                 line=dict(color=RED, width=2), marker=dict(size=7)), 1, 1)
    fig.add_hline(y=limit, line=dict(color=RED, dash="dot", width=1.2), row=1, col=1, annotation_text=f"limit {limit:.0f} °C", annotation_position="top right")
    fig.add_hline(y=pre, line=dict(color=ORANGE, dash="dot", width=1), row=1, col=1, annotation_text=f"pre-alarm {pre:.0f} °C", annotation_position="bottom right")
    fig.add_vline(x=now["t_min"], line=dict(color=NAVY, width=1))
    add_phase_bands(fig, run, row=1, col=1)
    fig.add_trace(go.Scatter(x=run["t_min"], y=run["hsd_kw_phys"], name="physics kW", line=dict(color=GREY, width=1.2, dash="dot")), 2, 1)
    fig.add_trace(go.Scatter(x=past["t_min"], y=past["hsd_kw"], name="HSD kW", line=dict(color=NAVY, width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=past["t_min"], y=past["hsd_rpm"], name="HSD rpm", line=dict(color=GREEN, width=1.2)), 2, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=past["t_min"], y=past["energy_cum_kwh"], name="kWh", line=dict(color=NAVY, width=2)), 3, 1)
    fig.add_trace(go.Scatter(x=past["t_min"], y=past["specific_energy_cum_kwh_t"], name="kWh/t", line=dict(color=ORANGE, width=1.5, dash="dash")), 3, 1, secondary_y=True)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="rpm", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="kWh", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="kWh/t", row=3, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="minutes from batch start", row=3, col=1, range=[0, run["t_min"].iloc[-1] + 1])
    chart_ph.plotly_chart(plot_layout(fig, height=720), use_container_width=True)
    foot_ph.caption(f"Twin so far: now-cast MAE {(past['T_meas'] - past['T_nowcast']).abs().mean():.2f} °C, 5-min-ahead MAE "
                    f"{(past['T_meas'] - past['T_ahead']).abs().mean():.2f} °C | indicators: HSD power {now['kw_resid_pct']:+.1f} % vs normal, "
                    f"temperature {now['T_resid_replay']:+.1f} °C vs normal gap, flow {100 * now['flow_ratio']:.0f} % of set-point, "
                    f"mass {100 * now['mass_ratio']:.0f} % of nominal")


render(i)
if playing:
    j = i
    while j < n - 1:
        j = min(j + speed, n - 1)
        st.session_state.t_idx = j
        render(j)
        time.sleep(0.12)
