"""
check_step3.py - generates the simulated plant year and proves it is fit for the twin.

What it does:
  1. generates the dataset into data/raw/ (deterministic, ~15 s) and reads it back
  2. structure: schema columns, 10-second resolution, step sequence, NaN rules
  3. physics plausibility: power within motor rating, jacket outlet above inlet, flows in range
  4. seasonality: chilled-water inlet and peak temperature by month
  5. noise layer: measured minus true temperature and power
  6. lab readings: grind-gauge checks per batch, extension logic
  7. faults: prevalence, blade-wear sawtooth, fouling episodes, and their visible effects
  8. overview plot -> reports/step3_dataset_overview.html

Run from the project folder:   python scripts/check_step3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import schema  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402
from tsd_twin.simulate import generate_dataset  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
RAW = ROOT / "data" / "raw"


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main() -> None:
    cfg = load_config()
    (ROOT / "reports").mkdir(exist_ok=True)

    # ------------------------------------------------------------------- 1
    banner("1. GENERATE")
    log = generate_dataset(cfg, RAW)
    ts = pd.read_csv(RAW / "timeseries.csv", parse_dates=["timestamp"])
    truth = pd.read_csv(RAW / "timeseries_truth.csv")
    b = pd.read_csv(RAW / "batches.csv", parse_dates=["start_time"])
    print(f"  {log['n_batches']} batches, {log['rows']:,} rows at 10 s, seed {log['seed']}")
    print(f"  calendar {b['start_time'].min():%Y-%m-%d} .. {b['start_time'].max():%Y-%m-%d}; "
          f"families {b['product_family'].value_counts().to_dict()}")
    print("  files: " + ", ".join(f"{f.name} ({f.stat().st_size / 1e6:.1f} MB)" for f in sorted(RAW.glob('*.csv'))))
    assert len(b) >= get(cfg, "simulation.n_batches"), "fewer batches than the required minimum"

    # ------------------------------------------------------------------- 2
    banner("2. STRUCTURE")
    assert list(ts.columns) == schema.measured_columns(), "timeseries columns differ from schema"
    assert list(truth.columns) == schema.truth_columns(), "truth columns differ from schema"
    missing = [c for c in schema.batch_columns() if c not in b.columns]
    assert not missing, f"batch table missing {missing}"
    print("  columns match schema.py (timeseries, truth, batches)             OK")
    dt = ts.groupby("batch_id")["t_s"].diff().dropna()
    assert (dt == 10).all(), "not every sample is 10 s apart"
    assert (ts.groupby("batch_id")["t_s"].first() == 0).all()
    print("  every batch starts at t=0 and every sample is exactly 10 s apart  OK")
    order = ts.groupby("batch_id")["step"].apply(lambda s: list(dict.fromkeys(s)))
    assert all(o == schema.STEP_NAMES for o in order), "step sequence broken"
    print("  every batch runs charge -> pigment -> high-speed -> letdown       OK")
    dense = [c for c in schema.measured_columns() if c not in ("hegman", "viscosity_ku")]
    assert not ts[dense].isna().any().any(), "NaN in a dense measured column"
    heg_per_batch = ts.groupby("batch_id")["hegman"].count()
    visc_per_batch = ts.groupby("batch_id")["viscosity_ku"].count()
    assert (heg_per_batch >= 1).all() and (visc_per_batch == 1).all()
    print(f"  no NaN in dense tags; Hegman readings per batch {heg_per_batch.min()}-{heg_per_batch.max()}, "
          f"one viscosity per batch                                          OK")
    rows_per_batch = ts.groupby("batch_id").size()
    print(f"  rows per batch: median {rows_per_batch.median():.0f} ({rows_per_batch.min()}-{rows_per_batch.max()}) "
          f"= {rows_per_batch.median() / 6:.0f} min typical cycle")

    # ------------------------------------------------------------------- 3
    banner("3. PHYSICAL PLAUSIBILITY OF THE MEASURED TAGS")
    rated = get(cfg, "equipment.hsd.motor_rated_kw")
    print(f"  HSD kW: max {ts['hsd_kw'].max():.1f} (rated {rated}); anchor kW max {ts['anchor_kw'].max():.1f}")
    assert ts["hsd_kw"].max() <= rated * 1.06, "HSD power exceeds rating by more than sensor noise"
    hsd = ts[ts["step"] == "high_speed_dispersion"]
    print(f"  in high-speed dispersion: rpm {hsd['hsd_rpm'].min():.0f}-{hsd['hsd_rpm'].max():.0f}, "
          f"kW {hsd['hsd_kw'].quantile(0.05):.1f}-{hsd['hsd_kw'].quantile(0.95):.1f} (5-95 %), "
          f"product T {hsd['T_product_c'].min():.1f}-{hsd['T_product_c'].max():.1f} degC")
    dT_w = hsd["T_jacket_out_c"] - hsd["T_jacket_in_c"]
    print(f"  jacket water rise in dispersion: median {dT_w.median():.2f} K, 95 % {dT_w.quantile(0.95):.2f} K")
    assert dT_w.median() > 0.5
    lo, hi = get(cfg, "equipment.cooling_jacket.flow_range_m3h")
    print(f"  chilled-water flow {ts['cw_flow_m3h'].min():.1f}-{ts['cw_flow_m3h'].max():.1f} m3/h "
          f"(set-points {sorted(float(x) for x in b['cw_flow_setpoint_m3h'].unique())}; below {lo} only in low-flow faults)")
    sag = b["hsd_rpm_setpoint"] - hsd.groupby("batch_id")["hsd_rpm"].min().reindex(b["batch_id"]).values
    print(f"  batches where actual rpm sagged > 20 rpm below set-point (drive limit): {(sag > 20).sum()}")

    # ------------------------------------------------------------------- 4
    banner("4. SEASONALITY")
    season = b.groupby("month").agg(cw_in=("T_cw_in_mean_c", "mean"), ambient=("T_ambient_c", "mean"),
                                    peak_T=("peak_temp_c", "mean"), peak_T_max=("peak_temp_c", "max"),
                                    breaches=("breached", "sum"), batches=("batch_id", "count")).round(1)
    print(season.T.to_string())
    assert season.loc[7, "cw_in"] - season.loc[1, "cw_in"] > 8, "summer/winter chilled-water difference too small"
    r = np.corrcoef(b["T_cw_in_mean_c"], b["peak_temp_c"])[0, 1]
    print(f"  correlation chilled-water inlet vs peak product T: {r:.2f}   (expected strongly positive)")
    assert r > 0.5
    print(f"  limit breaches: {int(b['breached'].sum())} of {len(b)} batches "
          f"({100 * b['breached'].mean():.1f} %), in months {sorted(int(x) for x in b.loc[b['breached'], 'month'].unique())}")
    print(f"  pre-alarm (within 3 degC of limit): {int(((b['peak_temp_c'] >= 42) & (b['product_family'] == 'water_based')).sum())} water-based batches")

    # ------------------------------------------------------------------- 5
    banner("5. NOISE LAYER (measured minus true)")
    m = ts[["batch_id", "t_s", "T_product_c", "hsd_kw"]].merge(truth[["batch_id", "t_s", "T_true_c", "hsd_kw_true"]],
                                                                on=["batch_id", "t_s"])
    err_T = m["T_product_c"] - m["T_true_c"]
    per_batch_offset = err_T.groupby(m["batch_id"]).mean()
    print(f"  temperature: within-batch noise sd {err_T.groupby(m['batch_id']).std().median():.3f} degC, "
          f"per-batch offset sd {per_batch_offset.std():.3f} degC")
    kw_rel = ((m["hsd_kw"] - m["hsd_kw_true"]) / m["hsd_kw_true"].clip(lower=1)).loc[m["hsd_kw_true"] > 10]
    print(f"  HSD power: relative noise sd {100 * kw_rel.std():.2f} % (set {100 * get(cfg, 'simulation.sensors.kw_noise_frac'):.1f} %)")
    e_err = 100 * (b["energy_total_kwh"] - truth.groupby("batch_id")["energy_cum_kwh"].last().reindex(b["batch_id"]).values) \
        / truth.groupby("batch_id")["energy_cum_kwh"].last().reindex(b["batch_id"]).values
    print(f"  batch energy from noisy kW vs exact: mean {e_err.mean():+.2f} %, sd {e_err.std():.2f} %")
    assert abs(e_err).max() < 2.0

    # ------------------------------------------------------------------- 6
    banner("6. LAB CHECKS AND THE DISPERSION STOP RULE")
    readings = ts["hegman"].dropna()
    assert np.allclose(readings * 2, np.round(readings * 2)), "Hegman readings must be multiples of 0.5"
    print(f"  {len(readings)} grind-gauge readings, all on the 0.5 scale; per batch "
          f"{heg_per_batch.describe()[['min', '50%', 'max']].astype(int).to_dict()}")
    ext = b["n_extensions"].value_counts().sort_index()
    print(f"  extensions after a failed check: {ext.to_dict()}  -> {100 * (b['n_extensions'] == 0).mean():.0f} % "
          f"of batches finish at the planned time")
    print(f"  target reached on a lab reading: {100 * b['hegman_reached_lab'].mean():.1f} % of batches; "
          f"dispersion time {b['dispersion_min'].min():.0f}-{b['dispersion_min'].max():.0f} min "
          f"(planned {b['planned_dispersion_min'].min():.0f}-{b['planned_dispersion_min'].max():.0f})")
    lab_vs_true = (b["time_to_target_hegman_lab_min"] - b["time_to_target_hegman_true_min"]).dropna()
    print(f"  lab time-to-target minus true crossing: median {lab_vs_true.median():+.1f} min "
          f"(lab can only see it at the next check)")
    print("\n  by recipe:")
    print(b.groupby("recipe_id").agg(n=("batch_id", "count"), mass_t=("batch_mass_kg", lambda s: s.mean() / 1000),
                                     kwh_t=("specific_energy_kwh_per_t", "mean"), disp_min=("dispersion_min", "mean"),
                                     ext=("n_extensions", "mean"), reached=("hegman_reached_lab", "mean"),
                                     peak_T=("peak_temp_c", "mean"), load_pct=("hsd_load_peak_pct", "max")).round(2).to_string())

    # ------------------------------------------------------------------- 7
    banner("7. INJECTED FAULTS (ground truth - simulator only)")
    counts = {f: int(b["faults_active"].fillna("").str.contains(f).sum())
              for f in ["BLADE_WEAR", "JACKET_FOULING", "LOW_CW_FLOW", "OVERCHARGE"]}
    print(f"  batches with each fault active (can overlap): {counts}; no fault: {int((b['fault_injected'] == 'NONE').sum())}")
    life = get(cfg, "simulation.faults.blade_life_batches")
    resets = [i for i in range(1, len(b)) if b["blade_wear"].iloc[i] < b["blade_wear"].iloc[i - 1] - 0.1]
    print(f"  blade replaced before batches {[b['batch_id'].iloc[i] for i in resets]} (life {life} batches)")
    assert len(resets) == len(b) // life
    wear_corr = np.corrcoef(b["blade_wear"], b["dispersion_min"])[0, 1]
    print(f"  correlation blade wear vs dispersion time: {wear_corr:.2f}  (worn blade -> longer grind)")
    assert wear_corr > 0.15
    fo = b[["batch_id", "jacket_fouling_m2k_w"]]
    episodes = ((fo["jacket_fouling_m2k_w"].diff() > 0.0008).sum())
    print(f"  fouling: base build-up to {b['jacket_fouling_m2k_w'].quantile(0.5):.5f} m2K/W median, "
          f"{episodes} dirty-water episode(s) start, max {b['jacket_fouling_m2k_w'].max():.5f}")
    print("\n  effect of each fault on the batch (means):")
    print(b.groupby("fault_injected").agg(n=("batch_id", "count"), peak_T=("peak_temp_c", "mean"),
                                          breaches=("breached", "sum"), disp_min=("dispersion_min", "mean"),
                                          ext=("n_extensions", "mean"), kwh_t=("specific_energy_kwh_per_t", "mean"),
                                          load_pct=("hsd_load_peak_pct", "mean")).round(2).to_string())
    print("  -> these differences are what Step 8 has to pick out of the measured tags alone.")

    # ------------------------------------------------------------------- 8
    banner("8. OVERVIEW PLOT")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed - plot skipped, everything else passed.")
        banner("ALL STEP 3 CHECKS PASSED - ready for Step 4 (features)")
        return
    navy, orange, grey = "#1F3864", "#C55A11", "#8A8A8A"
    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.10, horizontal_spacing=0.08,
                        subplot_titles=("Chilled-water inlet and hall temperature over the year",
                                        "Peak product temperature by month (water-based limit 45, solvent 55)",
                                        "Specific energy by recipe (kWh/t)",
                                        "Dispersion time vs blade wear",
                                        "Jacket fouling over the year, with limit breaches marked",
                                        "One batch: measured vs true temperature, lab Hegman vs true grind"))
    fig.add_trace(go.Scatter(x=b["start_time"], y=b["T_cw_in_mean_c"], mode="markers", name="CW inlet", marker=dict(color=navy, size=5)), 1, 1)
    fig.add_trace(go.Scatter(x=b["start_time"], y=b["T_ambient_c"], mode="markers", name="hall", marker=dict(color=orange, size=5)), 1, 1)
    for fam, col in [("water_based", navy), ("solvent_based", orange)]:
        sub = b[b["product_family"] == fam]
        fig.add_trace(go.Box(x=sub["month"], y=sub["peak_temp_c"], name=fam, marker_color=col, boxpoints=False), 1, 2)
    fig.add_hline(y=45, line=dict(color=navy, dash="dot", width=1), row=1, col=2)
    fig.add_hline(y=55, line=dict(color=orange, dash="dot", width=1), row=1, col=2)
    for fam, col in [("water_based", navy), ("solvent_based", orange)]:
        sub = b[b["product_family"] == fam]
        fig.add_trace(go.Box(x=sub["recipe_id"], y=sub["specific_energy_kwh_per_t"], name=fam, marker_color=col, showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=sub["blade_wear"], y=sub["dispersion_min"], mode="markers", name=fam, showlegend=False,
                                 marker=dict(color=col, size=6, opacity=0.7)), 2, 2)
    fig.add_trace(go.Scatter(x=b["start_time"], y=b["jacket_fouling_m2k_w"], mode="lines", name="fouling R", line=dict(color=grey)), 3, 1)
    br = b[b["breached"]]
    fig.add_trace(go.Scatter(x=br["start_time"], y=br["jacket_fouling_m2k_w"], mode="markers", name="limit breach",
                             marker=dict(color="#C00000", size=9, symbol="x")), 3, 1)
    ex_id = b.loc[b["breached"], "batch_id"].iloc[0] if b["breached"].any() else b["batch_id"].iloc[10]
    ex = ts[ts["batch_id"] == ex_id]
    ext = truth[truth["batch_id"] == ex_id]
    fig.add_trace(go.Scatter(x=ex["t_s"] / 60, y=ex["T_product_c"], name=f"{ex_id} measured T", line=dict(color=navy, width=1)), 3, 2)
    fig.add_trace(go.Scatter(x=ext["t_s"] / 60, y=ext["T_true_c"], name="true T", line=dict(color=orange, width=2, dash="dash")), 3, 2)
    labs = ex.dropna(subset=["hegman"])
    fig.add_trace(go.Scatter(x=labs["t_s"] / 60, y=labs["hegman"] * 5, mode="markers", name="lab Hegman x5",
                             marker=dict(color="#548235", size=9)), 3, 2)
    fig.add_trace(go.Scatter(x=ext["t_s"] / 60, y=ext["hegman_true"] * 5, name="true grind x5", line=dict(color="#548235", width=1, dash="dot")), 3, 2)
    fig.update_layout(title="Simulated plant year - dataset overview (SIMULATED, not plant data)", template="plotly_white",
                      height=1050, font=dict(family="Arial"), legend=dict(orientation="h", y=-0.04))
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="degC", row=1, col=2)
    fig.update_yaxes(title_text="kWh/t", row=2, col=1)
    fig.update_yaxes(title_text="min", row=2, col=2)
    fig.update_xaxes(title_text="blade wear (0 = new)", row=2, col=2)
    fig.update_yaxes(title_text="m2K/W", row=3, col=1)
    fig.update_yaxes(title_text="degC / Hegman x5", row=3, col=2)
    fig.update_xaxes(title_text="minutes", row=3, col=2)
    out = ROOT / "reports" / "step3_dataset_overview.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"  written {out.relative_to(ROOT)}")

    banner("ALL STEP 3 CHECKS PASSED - ready for Step 4 (features)")


if __name__ == "__main__":
    main()
