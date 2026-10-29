"""Page 4 - Alerts: predicted limit breaches several minutes ahead, with the likely cause."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import GREEN, GREY, NAVY, ORANGE, RED, add_phase_bands, batch_run, default_batch, fleet, header, inject_css, load_core, plot_layout, tiles

inject_css()
core = load_core()
cfg, batches, thr = core["cfg"], core["batches"], core["thr"]
header("Alerts", "Temperature pre-alarms raised ahead of time, their predicted cause, and the machine-condition flags")
V, A = fleet()

with st.sidebar:
    st.markdown("#### Alert filters")
    kinds = st.multiselect("Alert type", ["TEMP_PREALARM", "TEMP_LIMIT", "LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"],
                           default=["TEMP_PREALARM", "TEMP_LIMIT", "LOW_CW_FLOW", "OVERCHARGE"])
    only_first = st.checkbox("First episode per batch and type", value=True)
    st.markdown("#### Batch detail")
    ids = list(batches.index)
    dflt = default_batch()
    detail_id = st.selectbox("Batch", ids, index=ids.index(dflt))

# ---------------------------------------------------------------- summary --
temp = A[A["alert"] == "TEMP_PREALARM"] if not A.empty else pd.DataFrame()
if not temp.empty and only_first:
    temp = temp.sort_values("raised_at_min").drop_duplicates(["batch_id"])
true_pre = temp[temp["limit_crossed_later"] == True] if not temp.empty else pd.DataFrame()  # noqa: E712
n_breach = int(V["breached"].sum())
caught = int(V.loc[V["breached"], "prealarm_lead_min"].notna().sum())
tiles([
    ("Limit breaches", n_breach, "", "in the simulated year", "alarm" if n_breach else "ok"),
    ("Pre-alarmed in advance", f"{caught}/{n_breach}", "", "breaches with a pre-alarm before the crossing", "ok" if caught == n_breach else "warn"),
    ("Lead time", "-" if V["prealarm_lead_min"].dropna().empty else f"{V['prealarm_lead_min'].median():.0f}", "min",
     f"median (min {V['prealarm_lead_min'].min():.0f}); target >= {cfg['kpi_targets']['breach_lead_time_min']['value']}" if V["prealarm_lead_min"].notna().any() else ""),
    ("Watch flags", int(((V["temp_flag"] == "PREALARM") & ~V["breached"]).sum()), "", "pre-alarms on batches that stayed under the limit", "warn"),
    ("Machine flags", int((V["flags"] != "NONE").sum()), "", "batches with wear / fouling / flow / overcharge flags"),
])

# ----------------------------------------------------------- alert table --
st.markdown("##### Alert log")
if A.empty:
    st.info("No alerts in the history.")
else:
    L = A[A["alert"].isin(kinds)].copy()
    if only_first:
        L = L.sort_values("raised_at_min").drop_duplicates(["batch_id", "alert"])
    L = L.merge(V[["batch_id", "start_time", "recipe_id"]], on="batch_id", how="left").sort_values(["start_time", "raised_at_min"], ascending=[False, True])
    L["start_time"] = pd.to_datetime(L["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
    L = L[["batch_id", "start_time", "recipe_id", "alert", "raised_at_min", "step", "T_now_c", "T_next5_c", "breach_eta_min", "lead_time_min",
           "limit_crossed_later", "likely_cause"]]
    L.columns = ["Batch", "Start", "Recipe", "Alert", "Raised at (min)", "Phase", "T now (°C)", "T +5 min (°C)", "Breach ETA (min)",
                 "Actual lead (min)", "Limit crossed", "Likely cause / evidence"]
    st.dataframe(L, use_container_width=True, hide_index=True, height=380)
    st.caption("'Breach ETA' is the projection at the moment the alert was raised; 'Actual lead' is how long before the real crossing it came. "
               "Blank = the limit was never crossed (a watch flag).")

# ------------------------------------------------------------ batch detail --
st.markdown(f"##### Batch detail - {detail_id}: how the warning developed")
run = batch_run(detail_id)
meta = batches.loc[detail_id]
limit, pre = run["limit_c"].iloc[0], run["prealarm_c"].iloc[0]
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.38, 0.2, 0.22, 0.2],
                    subplot_titles=("Measured temperature, 5-min-ahead prediction (shifted to its target time) and projected peak",
                                    "Breach ETA (min) as it evolved", "Machine indicators vs their normal level (HSD power %, jacket U drop %)", "Alert state"))
fig.add_trace(go.Scatter(x=run["t_min"], y=run["T_meas"], name="measured", line=dict(color=NAVY, width=2.2)), 1, 1)
fig.add_trace(go.Scatter(x=run["t_min"], y=run["T_ahead"], name="predicted 5 min earlier", line=dict(color=RED, width=1.5, dash="dash")), 1, 1)
fig.add_trace(go.Scatter(x=run["t_min"], y=run["T_peak_projected"], name="projected peak", line=dict(color=ORANGE, width=1.2, dash="dot")), 1, 1)
fig.add_hline(y=limit, line=dict(color=RED, dash="dot"), row=1, col=1, annotation_text=f"limit {limit:.0f} °C")
fig.add_hline(y=pre, line=dict(color=ORANGE, dash="dot", width=1), row=1, col=1, annotation_text=f"pre-alarm {pre:.0f} °C")
add_phase_bands(fig, run, row=1, col=1)
fig.add_trace(go.Scatter(x=run["t_min"], y=run["breach_eta_min"], name="breach ETA", line=dict(color=RED, width=2)), 2, 1)
fig.add_hline(y=thr.horizon_min, line=dict(color=GREY, dash="dot", width=1), row=2, col=1, annotation_text=f"{thr.horizon_min:.0f} min")
fig.add_trace(go.Scatter(x=run["t_min"], y=run["kw_resid_pct"], name="HSD power vs normal (%)", line=dict(color=NAVY, width=1.5)), 3, 1)
fig.add_trace(go.Scatter(x=run["t_min"], y=run["u_drop_pct"], name="jacket U drop vs normal (%)", line=dict(color=ORANGE, width=1.5)), 3, 1)
fig.add_hline(y=thr.blade_wear_kw_pct, line=dict(color=NAVY, dash="dot", width=1), row=3, col=1)
fig.add_hline(y=thr.fouling_u_drop_pct, line=dict(color=ORANGE, dash="dot", width=1), row=3, col=1)
codes = ["TEMP_LIMIT", "TEMP_PREALARM", "LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"]
cols = {"TEMP_LIMIT": RED, "TEMP_PREALARM": ORANGE, "LOW_CW_FLOW": "#2E75B6", "OVERCHARGE": "#7030A0", "JACKET_FOULING": GREEN, "BLADE_WEAR": GREY}
for k, c in enumerate(codes):
    on = run[f"rule_{c}"].values
    fig.add_trace(go.Scatter(x=run["t_min"], y=np.where(on, len(codes) - k, np.nan), name=c, mode="markers",
                             marker=dict(color=cols[c], size=6, symbol="square")), 4, 1)
fig.update_yaxes(title_text="°C", row=1, col=1)
fig.update_yaxes(title_text="min", row=2, col=1)
fig.update_yaxes(title_text="%", row=3, col=1)
fig.update_yaxes(tickvals=list(range(len(codes), 0, -1)), ticktext=codes, row=4, col=1)
fig.update_xaxes(title_text="minutes from batch start", row=4, col=1)
st.plotly_chart(plot_layout(fig, height=860), use_container_width=True)

ad = A[A["batch_id"] == detail_id] if not A.empty else pd.DataFrame()
if ad.empty:
    st.success("No alerts on this batch.")
else:
    st.dataframe(ad[["alert", "raised_at_min", "step", "duration_min", "T_now_c", "T_next5_c", "breach_eta_min", "lead_time_min", "limit_crossed_later", "likely_cause"]]
                 .rename(columns={"alert": "Alert", "raised_at_min": "Raised at (min)", "step": "Phase", "duration_min": "Held (min)", "T_now_c": "T now (°C)",
                                  "T_next5_c": "T +5 min (°C)", "breach_eta_min": "Breach ETA (min)", "lead_time_min": "Actual lead (min)",
                                  "limit_crossed_later": "Limit crossed", "likely_cause": "Likely cause / evidence"}),
                 use_container_width=True, hide_index=True)
st.caption(f"Rules (config 'alerts'): pre-alarm when the 5-min-ahead temperature or the measured temperature exceeds limit - {limit - pre:.0f} °C; "
           f"blade wear when HSD power is {thr.blade_wear_kw_pct:.0f} % below its normal level for {thr.hold_rows * 10 / 60:.0f} min in high-speed dispersion; "
           f"jacket fouling (rule variant '{thr.fouling_mode}') when the implied U from the measured jacket duty is {thr.fouling_u_drop_pct:.1f} % below its normal level "
           f"(or the product runs {thr.fouling_T_resid_c:.2f} °C above its normal gap to the measured-power physics) with normal flow; "
           f"low flow below {100 * thr.low_flow_ratio:.0f} % of set-point; overcharge above {100 * thr.overcharge_ratio:.0f} % of nominal mass. "
           f"Normal levels: HSD power {core['baselines'].kw_resid_pct:+.1f} %, temperature gap {core['baselines'].T_resid_replay_c:+.1f} °C "
           f"(learned from {core['baselines'].n_batches} batches).")
