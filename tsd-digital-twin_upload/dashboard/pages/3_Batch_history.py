"""Page 3 - Batch history: KPI trends, energy league table by product, fault flags."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import GREEN, GREY, NAVY, ORANGE, PURPLE, RED, fleet, header, inject_css, load_core, plot_layout, tiles
from tsd_twin import faults as fl

inject_css()
core = load_core()
cfg = core["cfg"]
header("Batch history", "KPI trends, energy league table by product, and the twin's fault flags over the simulated year")
V, A = fleet()
V = V.sort_values("start_time")

with st.sidebar:
    st.markdown("#### Filters")
    fam = st.multiselect("Family", ["water_based", "solvent_based"], default=["water_based", "solvent_based"])
    recs = st.multiselect("Recipe", sorted(V["recipe_id"].unique()), default=[])
    d0, d1 = V["start_time"].min().date(), V["start_time"].max().date()
    dr = st.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
    show_truth = st.checkbox("Show simulator ground truth (validation only)", value=False)
    window = st.slider("Rolling window (batches)", 5, 40, 15)

m = V["product_family"].isin(fam)
if recs:
    m &= V["recipe_id"].isin(recs)
if isinstance(dr, tuple) and len(dr) == 2:
    m &= (V["start_time"].dt.date >= dr[0]) & (V["start_time"].dt.date <= dr[1])
H = V[m].copy()
if H.empty:
    st.warning("No batches match the filters.")
    st.stop()

lim = np.where(H["product_family"] == "water_based", 45, 55)
H["margin"] = lim - H["peak_T"]
tiles([
    ("Batches", len(H), "", f"{H['start_time'].min():%d %b %Y} - {H['start_time'].max():%d %b %Y}"),
    ("Specific energy", f"{H['kwh_per_t'].median():.2f}", "kWh/t", f"P90 {H['kwh_per_t'].quantile(0.9):.2f}"),
    ("Limit breaches", int(H["breached"].sum()), "", f"{100 * H['breached'].mean():.1f} % of batches", "alarm" if H["breached"].any() else "ok"),
    ("Pre-alarm lead", "-" if H["prealarm_lead_min"].dropna().empty else f"{H['prealarm_lead_min'].median():.0f}", "min",
     "median warning before a breach"),
    ("Extensions", f"{100 * (H['n_extensions'] > 0).mean():.0f}", "%", "batches needing a longer grind"),
    ("Target reached", f"{100 * H['hegman_reached_lab'].mean():.0f}", "%", "on a lab reading"),
    ("Twin accuracy", f"{H['nowcast_mae'].median():.2f}", "°C", f"median now-cast MAE; 5-min-ahead {H['ahead_mae'].median():.2f} °C"),
])

# ------------------------------------------------------------------ trends --
fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.07,
                    subplot_titles=("Specific energy (kWh/t)", "Peak product temperature and margin to limit (°C)",
                                    "Dispersion time (min) and extensions", "Twin fault indicators (vs their normal level)"),
                    specs=[[{}, {}], [{"secondary_y": True}, {}]])
col = np.where(H["product_family"] == "water_based", NAVY, ORANGE)
fig.add_trace(go.Scatter(x=H["start_time"], y=H["kwh_per_t"], mode="markers", name="kWh/t", marker=dict(color=col, size=6, opacity=0.7)), 1, 1)
fig.add_trace(go.Scatter(x=H["start_time"], y=H["kwh_per_t"].rolling(window, min_periods=3).median(), name=f"rolling median ({window})",
                         line=dict(color=GREY, width=2)), 1, 1)
fig.add_trace(go.Scatter(x=H["start_time"], y=H["peak_T"], mode="markers", name="peak T", marker=dict(color=np.where(H["breached"], RED, col), size=6, opacity=0.8)), 1, 2)
fig.add_hline(y=45, line=dict(color=NAVY, dash="dot", width=1), row=1, col=2, annotation_text="45 °C water-based")
fig.add_hline(y=55, line=dict(color=ORANGE, dash="dot", width=1), row=1, col=2, annotation_text="55 °C solvent-based")
fig.add_trace(go.Scatter(x=H["start_time"], y=H["dispersion_min"], mode="markers", name="dispersion min",
                         marker=dict(color=np.where(H["n_extensions"] > 0, RED, col), size=6, opacity=0.7)), 2, 1)
fig.add_trace(go.Scatter(x=H["start_time"], y=100 * (H["n_extensions"] > 0).rolling(window, min_periods=3).mean(), name="% with extensions (rolling)",
                         line=dict(color=GREY, width=2)), 2, 1, secondary_y=True)
fig.add_trace(go.Scatter(x=H["start_time"], y=H["kw_resid_pct_hsd"], name="HSD power vs normal (%)", line=dict(color=NAVY, width=1.5)), 2, 2)
fig.add_trace(go.Scatter(x=H["start_time"], y=H["u_drop_pct_hsd"], name="jacket U drop vs normal (%)", line=dict(color=ORANGE, width=1.5)), 2, 2)
fig.add_hline(y=core["thr"].blade_wear_kw_pct, line=dict(color=NAVY, dash="dot", width=1), row=2, col=2)
fig.add_hline(y=core["thr"].fouling_u_drop_pct, line=dict(color=ORANGE, dash="dot", width=1), row=2, col=2)
if show_truth:
    fig.add_trace(go.Scatter(x=H["start_time"], y=-20 * core["batches"].loc[H["batch_id"], "blade_wear"].values, name="truth: -20 x blade wear",
                             line=dict(color=NAVY, width=1, dash="dash")), 2, 2)
    fig.add_trace(go.Scatter(x=H["start_time"], y=15000 * core["batches"].loc[H["batch_id"], "jacket_fouling_m2k_w"].values, name="truth: fouling x15000",
                             line=dict(color=ORANGE, width=1, dash="dash")), 2, 2)
fig.update_yaxes(title_text="kWh/t", row=1, col=1); fig.update_yaxes(title_text="°C", row=1, col=2)
fig.update_yaxes(title_text="min", row=2, col=1, secondary_y=False); fig.update_yaxes(title_text="% with extensions", row=2, col=1, secondary_y=True, showgrid=False, range=[0, 100])
fig.update_yaxes(title_text="% / °C", row=2, col=2)
st.plotly_chart(plot_layout(fig, height=640), use_container_width=True)
st.caption("Navy = water-based, orange = solvent-based, red = breach / extension. Dotted lines on the indicator chart are the alert thresholds.")

# ------------------------------------------------------------ drift tracker --
st.markdown("##### Slow drift across batches: blade wear and jacket fouling")
thr = core["thr"]
Hc = H.sort_values("start_time").reset_index(drop=True)
wear_tr = fl.drift_tracker(Hc["kw_resid_pct_hsd"], 0.0, "down", thr.drift_slack, thr.drift_h, thr.drift_window, thr.blade_wear_kw_pct)
foul_tr = fl.drift_tracker(Hc["u_drop_pct_hsd"], 0.0, "up", thr.drift_slack, thr.drift_h, thr.drift_window, thr.fouling_u_drop_pct)
last_w, last_f = wear_tr.iloc[-1], foul_tr.iloc[-1]
tiles([
    ("Blade: power vs normal", f"{Hc['kw_resid_pct_hsd'].iloc[-5:].median():+.1f}", "%", f"alarm at {thr.blade_wear_kw_pct:.1f} % (last 5 batches)",
     "alarm" if Hc["kw_resid_pct_hsd"].iloc[-5:].median() < thr.blade_wear_kw_pct else "ok"),
    ("Blade: trend", f"{last_w['slope_per_batch'] * 10:+.2f}" if pd.notna(last_w["slope_per_batch"]) else "-", "%/10 b",
     f"% per 10 batches, {thr.drift_window}-batch linear trend"),
    ("Blade: batches to alarm", f"{last_w['batches_to_alarm']:.0f}" if pd.notna(last_w["batches_to_alarm"]) else "-", "",
     "at the current trend (rough)", "warn" if pd.notna(last_w["batches_to_alarm"]) and last_w["batches_to_alarm"] < 20 else ""),
    ("Jacket: U drop vs normal", f"{Hc['u_drop_pct_hsd'].iloc[-5:].median():+.1f}", "%", f"alarm at {thr.fouling_u_drop_pct:.1f} % (last 5 batches)",
     "alarm" if Hc["u_drop_pct_hsd"].iloc[-5:].median() > thr.fouling_u_drop_pct else "ok"),
    ("Jacket: batches to alarm", f"{last_f['batches_to_alarm']:.0f}" if pd.notna(last_f["batches_to_alarm"]) else "-", "",
     "at the current trend (rough)", "warn" if pd.notna(last_f["batches_to_alarm"]) and last_f["batches_to_alarm"] < 20 else ""),
    ("Sustained-trend alarms", f"{int((wear_tr['cusum_alarm'] & ~wear_tr['cusum_alarm'].shift(1, fill_value=False)).sum())} / "
     f"{int((foul_tr['cusum_alarm'] & ~foul_tr['cusum_alarm'].shift(1, fill_value=False)).sum())}", "", "blade / jacket CUSUM episodes in the window"),
])
fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.09, subplot_titles=("Blade: HSD power vs normal, CUSUM and alarm episodes",
                                                                               "Jacket: implied-U drop vs normal, CUSUM and alarm episodes"),
                     specs=[[{"secondary_y": True}, {"secondary_y": True}]])
for col_i, (tr_, y, name, colr, level) in enumerate([(wear_tr, Hc["kw_resid_pct_hsd"], "power vs normal (%)", NAVY, thr.blade_wear_kw_pct),
                                                     (foul_tr, Hc["u_drop_pct_hsd"], "U drop vs normal (%)", ORANGE, thr.fouling_u_drop_pct)], start=1):
    fig2.add_trace(go.Scatter(x=Hc["start_time"], y=y, name=name, line=dict(color=colr, width=1.3)), 1, col_i)
    fig2.add_trace(go.Scatter(x=Hc["start_time"], y=tr_["cusum"], name="CUSUM (right axis)", line=dict(color=PURPLE, width=1)), 1, col_i, secondary_y=True)
    fig2.add_hline(y=thr.drift_h, line=dict(color=PURPLE, dash="dot", width=1), row=1, col=col_i, secondary_y=True)
    al = tr_["cusum_alarm"].values
    fig2.add_trace(go.Scatter(x=Hc.loc[al, "start_time"], y=y[al], mode="markers", name="sustained-trend alarm", marker=dict(color=RED, size=5)), 1, col_i)
    fig2.add_hline(y=level, line=dict(color=RED, dash="dot", width=1), row=1, col=col_i, annotation_text="per-batch alarm level")
fig2.update_yaxes(title_text="%", row=1, col=1, secondary_y=False); fig2.update_yaxes(title_text="%", row=1, col=2, secondary_y=False)
fig2.update_yaxes(title_text="CUSUM", row=1, col=1, secondary_y=True, showgrid=False); fig2.update_yaxes(title_text="CUSUM", row=1, col=2, secondary_y=True, showgrid=False)
st.plotly_chart(plot_layout(fig2, height=340), use_container_width=True)
st.caption("The CUSUM accumulates every batch's departure from normal beyond a small allowance and resets after a maintenance event "
           "(three batches back on the healthy side). The batches-to-alarm figure is a straight-line extrapolation of the last "
           f"{thr.drift_window} batches - use it to plan an inspection, not to schedule it to the day.")

# -------------------------------------------------------------- league table --
st.markdown("##### Energy league table by product")
lt = (H.groupby(["recipe_id", "recipe_name", "product_family"])
      .agg(batches=("batch_id", "size"), mass_t=("batch_mass_kg", lambda s: s.mean() / 1000), kwh_t_median=("kwh_per_t", "median"),
           kwh_t_best=("kwh_per_t", "min"), kwh_t_p90=("kwh_per_t", "quantile"), peak_T_median=("peak_T", "median"), breaches=("breached", "sum"),
           extensions_pct=("n_extensions", lambda s: 100 * (s > 0).mean()), dispersion_median=("dispersion_min", "median"))
      .reset_index().sort_values("kwh_t_median"))
lt["kwh_t_p90"] = H.groupby("recipe_id")["kwh_per_t"].quantile(0.9).reindex(lt["recipe_id"]).values
lt["saving_vs_best_pct"] = 100 * (lt["kwh_t_median"] - lt["kwh_t_best"]) / lt["kwh_t_median"]
lt = lt.round({"mass_t": 2, "kwh_t_median": 2, "kwh_t_best": 2, "kwh_t_p90": 2, "peak_T_median": 1, "extensions_pct": 0, "dispersion_median": 0, "saving_vs_best_pct": 0})
lt.columns = ["Recipe", "Product", "Family", "Batches", "Mass (t)", "kWh/t median", "kWh/t best", "kWh/t P90", "Peak T median (°C)", "Breaches",
              "Extensions (%)", "Dispersion median (min)", "Median vs best (%)"]
st.dataframe(lt, use_container_width=True, hide_index=True)
st.caption("'Median vs best' is the gap between a product's typical batch and its best batch - the first place to look for savings.")

# ---------------------------------------------------------------- fault flags --
st.markdown("##### Fault flags raised by the twin")
F = H[["batch_id", "start_time", "recipe_id", "flags", "temp_flag", "breached", "prealarm_lead_min", "kw_resid_pct_hsd", "u_drop_pct_hsd", "flow_ratio", "mass_ratio"]].copy()
if show_truth:
    F["simulator truth"] = core["batches"].loc[F["batch_id"], "faults_active"].fillna("").replace("", "NONE").values
only_flagged = st.checkbox("Only batches with a flag", value=True)
if only_flagged:
    F = F[(F["flags"] != "NONE") | (F["temp_flag"] != "NONE")]
F = F.sort_values("start_time", ascending=False)
F["start_time"] = F["start_time"].dt.strftime("%Y-%m-%d %H:%M")
F = F.round({"prealarm_lead_min": 0, "kw_resid_pct_hsd": 1, "u_drop_pct_hsd": 1, "flow_ratio": 2, "mass_ratio": 3})
F.columns = ["Batch", "Start", "Recipe", "Flags", "Temperature", "Breached", "Pre-alarm lead (min)", "HSD power vs normal (%)",
             "Jacket U drop vs normal (%)", "Flow / set-point", "Mass / nominal"] + (["Simulator truth"] if show_truth else [])
st.dataframe(F, use_container_width=True, hide_index=True, height=360)
if show_truth:
    both = H[(H["flags"] != "NONE")]
    tp = {}
    for k, name in [("BLADE", "BLADE_WEAR"), ("FOUL", "JACKET_FOULING"), ("FLOW", "LOW_CW_FLOW"), ("OVER", "OVERCHARGE")]:
        act = core["batches"].loc[H["batch_id"], "faults_active"].fillna("").str.contains(name).values
        flg = H["flags"].str.contains(k).values
        tp[name] = f"{int((act & flg).sum())}/{int(act.sum())} caught, {int((~act & flg).sum())} false"
    st.caption("Validation against the simulator's hidden faults - " + "; ".join(f"{k}: {v}" for k, v in tp.items()) + ". 'False' here counts every "
               "unlabelled batch, including the grey zone (wear 10-18 %, mild scale) where a flag is an early warning; see reports/step8_rule_cards.md.")
