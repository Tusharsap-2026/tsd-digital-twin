"""
check_step4.py - builds the feature tables and proves they are sound.

What it does:
  1. builds data/processed/ (row features with physics baselines, batch features, split)
  2. audits the feature tables: shapes, NaN pattern, feature groups
  3. audits the split: by batch, stratified, no leakage, time split, folds
  4. previews the physics-only baseline against the measured data (what Step 5 must beat)
  5. checks the lot / fault indicators carry signal (against the simulator's truth, reporting only)
  6. plot -> reports/step4_features_overview.html

Run from the project folder:   python scripts/check_step4.py     (needs data/raw from Step 3)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = Path(os.environ.get("TSD_DATA_DIR", ROOT / "data" / "raw"))
_PROC_DIR = ROOT / "data" / ("processed" if _RAW_DIR.name == "raw" else f"processed_{_RAW_DIR.name}")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import features as ft  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
RAW, PROC = _RAW_DIR, _PROC_DIR


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main() -> None:
    cfg = load_config()
    (ROOT / "reports").mkdir(exist_ok=True)
    if not (RAW / "timeseries.csv").exists():
        raise SystemExit("data/raw/timeseries.csv not found - run scripts/check_step3.py first")

    # ------------------------------------------------------------------- 1
    banner("1. BUILD FEATURES")
    t = time.time()
    out = ft.build_features(cfg, RAW, PROC)
    rf, bf, split = out["row_features"], out["batch_features"], out["split"]
    print(f"  built in {time.time() - t:.0f} s: row_features {rf.shape}, batch_features {bf.shape}, split {split.shape}")
    print("  files: " + ", ".join(f"{f.name} ({f.stat().st_size / 1e6:.1f} MB)" for f in sorted(PROC.glob('*.csv*'))))
    fd = ft.feature_dictionary()
    print("  feature groups:")
    for g, n in fd.groupby("group", sort=False).size().items():
        print(f"    {n:3d}  {g}")
    listed = set(fd["feature"])
    unlisted = [c for c in list(rf.columns) + list(bf.columns) if c not in listed]
    assert not unlisted, f"columns missing from the feature dictionary: {unlisted}"

    # ------------------------------------------------------------------- 2
    banner("2. NaN PATTERN (every NaN must have a reason)")
    nan_rows = rf.isna().sum()
    print("  row features with NaN:")
    for c, n in nan_rows[nan_rows > 0].items():
        print(f"    {c:28s} {n:7d} rows")
    lag_cols = [c for c in rf.columns if c.startswith("lag_")]
    per_batch_lag_nan = rf.groupby("batch_id")[lag_cols[0]].apply(lambda s: s.isna().sum())
    horizon_rows = int(get(cfg, "kpi_targets.breach_lead_time_min") * 60 / get(cfg, "data.sample_period_s"))
    assert (per_batch_lag_nan == horizon_rows).all(), "lag features must be NaN for exactly the first horizon rows"
    print(f"  -> lag_* are NaN for exactly the first {horizon_rows} rows of each batch (the {get(cfg, 'kpi_targets.breach_lead_time_min')} min horizon)")
    assert rf.loc[rf['step'] != 'charge_liquids', 'hegman_phys'].notna().all()
    print("  -> hegman_phys is NaN only before pigment goes in; hegman_lab is sparse by design")
    dense = [c for c in rf.columns if c not in lag_cols + ["hegman_phys", "hegman_lab"]]
    assert not rf[dense].isna().any().any(), "unexpected NaN in a dense row feature"
    nan_b = bf.isna().sum()
    print("  batch features with NaN: " + ", ".join(f"{c} ({n})" for c, n in nan_b[nan_b > 0].items()))
    print("  -> time-to-target is NaN when the target was never reached (lab) or not within the planned time (physics)")

    # ------------------------------------------------------------------- 3
    banner("3. SPLIT BY BATCH")
    assert split["batch_id"].is_unique
    for col in ["split_random", "split_time"]:
        tab = pd.crosstab(split["product_family"], split[col])
        print(f"  {col}:\n{tab.to_string()}")
    rows_test = rf["batch_id"].isin(split.loc[split["split_random"] == "test", "batch_id"])
    print(f"  random split: {rows_test.sum():,} test rows / {(~rows_test).sum():,} train rows; "
          f"test share of batches {100 * (split['split_random'] == 'test').mean():.0f} %")
    print(f"  time split: train ends {split.loc[split['split_time'] == 'train', 'start_time'].max():%Y-%m-%d}, "
          f"test starts {split.loc[split['split_time'] == 'test', 'start_time'].min():%Y-%m-%d}")
    print(f"  folds: {split['fold'].value_counts().sort_index().to_dict()}")
    overlap = set(rf.loc[rows_test, "batch_id"]) & set(rf.loc[~rows_test, "batch_id"])
    assert not overlap, "a batch appears in both train and test"
    print("  no batch appears on both sides of any split                        OK")

    # ------------------------------------------------------------------- 4
    banner("4. PHYSICS-ONLY PREVIEW (what the Step 5 models have to beat)")
    truth_b = pd.read_csv(RAW / "batches.csv").set_index("batch_id")      # labels for REPORTING only
    rf["fault"] = rf["batch_id"].map(truth_b["fault_injected"])
    rf["family"] = np.where(rf["family_wb"] == 1, "water_based", "solvent_based")
    rf["abs_open"] = rf["resid_open_c"].abs()
    rf["abs_replay"] = rf["resid_replay_c"].abs()
    per_batch = rf.groupby("batch_id").agg(mae_open=("abs_open", "mean"), mae_replay=("abs_replay", "mean"),
                                           max_open=("abs_open", "max"), bias_open=("resid_open_c", "mean"),
                                           bias_replay=("resid_replay_c", "mean"), fault=("fault", "first"), family=("family", "first"))
    tgt = get(cfg, "kpi_targets.temp_mae_c")
    print(f"  temperature, per-batch MAE (target {tgt} degC):")
    print(f"    physics open (modelled power):   median {per_batch['mae_open'].median():.2f}  P90 {per_batch['mae_open'].quantile(0.9):.2f}  "
          f"worst {per_batch['mae_open'].max():.2f}  pass {100 * (per_batch['mae_open'] <= tgt).mean():.0f} %  bias {per_batch['bias_open'].mean():+.2f}")
    print(f"    physics replay (measured power): median {per_batch['mae_replay'].median():.2f}  P90 {per_batch['mae_replay'].quantile(0.9):.2f}  "
          f"worst {per_batch['mae_replay'].max():.2f}  pass {100 * (per_batch['mae_replay'] <= tgt).mean():.0f} %  bias {per_batch['bias_replay'].mean():+.2f}")
    print("  by injected fault (truth labels, reporting only):")
    print(per_batch.groupby("fault").agg(n=("mae_open", "size"), mae_open=("mae_open", "median"), mae_replay=("mae_replay", "median"),
                                         bias_open=("bias_open", "mean"), bias_replay=("bias_replay", "mean")).round(2).to_string())
    print("  by family:")
    print(per_batch.groupby("family").agg(n=("mae_open", "size"), mae_open=("mae_open", "median"), mae_replay=("mae_replay", "median"),
                                          bias_open=("bias_open", "mean")).round(2).to_string())
    e_err = 100 * (bf["phys_energy_total_kwh"] - bf["y_energy_total_kwh"]) / bf["y_energy_total_kwh"]
    no_ext = bf["y_n_extensions"] == 0
    print(f"\n  energy, physics forecast for the planned schedule vs actual: all batches MAPE {e_err.abs().mean():.1f} %, "
          f"bias {e_err.mean():+.1f} %; batches without extensions MAPE {e_err[no_ext].abs().mean():.1f} %, "
          f"with extensions {e_err[~no_ext].abs().mean():.1f} %")
    t_err = bf["phys_time_to_target_min"] - bf["y_time_to_target_lab_min"]
    print(f"  time to target Hegman, physics vs lab: median {t_err.median():+.1f} min, MAE {t_err.abs().mean():.1f} min, "
          f"physics says 'not within planned time' for {bf['phys_time_to_target_min'].isna().sum()} batches "
          f"(lab never reached: {bf['y_time_to_target_lab_min'].isna().sum()})")
    pk = (bf["phys_peak_temp_c"] - bf["y_peak_temp_c"])
    print(f"  peak temperature, physics vs measured: bias {pk.mean():+.2f} degC, MAE {pk.abs().mean():.2f} degC; "
          f"breach predicted/actual: {int(bf['phys_breached'].sum())}/{int(bf['y_breached'].sum())}")
    print("  -> the open-loop physics is biased because the simulated plant deliberately differs from the twin's\n"
          "     parameters (simulation.plant_mismatch). That bias is what Step 5's residual model has to learn.")

    # ------------------------------------------------------------------- 5
    banner("5. DO THE EVIDENCE FEATURES CARRY SIGNAL?  (correlations against hidden truth)")
    bt = bf.set_index("batch_id").join(truth_b[["viscosity_factor", "grind_energy_factor", "blade_wear", "jacket_fouling_m2k_w", "cw_flow_actual_frac"]])
    checks = [
        ("pigment_kw_resid_phys_pct", "viscosity_factor", "HSD kW surplus at a known speed during pigment addition <-> viscosity lot"),
        ("pigment_anchor_kw_resid_pct", "viscosity_factor", "anchor kW surplus during pigment addition              <-> viscosity lot"),
        ("hsd5_kw_resid_phys_pct", "blade_wear", "kW shortfall in the first 5 min of dispersion           <-> blade wear"),
        ("hsd5_T_resid_replay_c", "jacket_fouling_m2k_w", "temperature above physics after 5 min (measured power) <-> fouling"),
        ("hsd5_cooling_duty_kw", "cw_flow_actual_frac", "measured cooling duty                                   <-> flow fraction"),
    ]
    for feat, tru, text in checks:
        r = bt[[feat, tru]].corr().iloc[0, 1]
        print(f"  r = {r:+.2f}   {text}")
    r_visc = max(bt[["pigment_kw_resid_phys_pct", "viscosity_factor"]].corr().iloc[0, 1],
                 bt[["pigment_anchor_kw_resid_pct", "viscosity_factor"]].corr().iloc[0, 1])
    r_wear = bt[["hsd5_kw_resid_phys_pct", "blade_wear"]].corr().iloc[0, 1]
    r_fo = bt[["hsd5_T_resid_replay_c", "jacket_fouling_m2k_w"]].corr().iloc[0, 1]
    assert r_fo > 0.5 and r_wear < -0.5, "fouling / wear indicators carry no signal"
    print(f"  -> blade wear shows up strongly as missing kW at a given speed (r = {r_wear:.2f}); fouling leaves a\n"
          f"     temperature residual even when the measured power is used (r = {r_fo:.2f}).")
    if r_visc < 0.5:
        print(f"     The viscosity lot is NOT readable from motor power (best r = {r_visc:.2f}): at Re 2,000-8,000 the power\n"
              f"     number barely moves with viscosity. Lot effects will stay as unexplained residual unless an inline\n"
              f"     viscometer or a low-speed torque check is added - a finding to carry into Step 9.")

    # ------------------------------------------------------------------- 6
    banner("6. PLOT")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed - plot skipped, everything else passed.")
        banner("ALL STEP 4 CHECKS PASSED - ready for Step 5 (models)")
        return
    navy, orange, grey, green, red = "#1F3864", "#C55A11", "#8A8A8A", "#548235", "#C00000"
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=("Measured vs physics temperature: a clean, a fouled and a breached batch",
                                        "Per-batch temperature residual of the open-loop physics, by injected fault",
                                        "Viscosity-lot indicator: anchor kW residual during pigment addition vs true lot factor",
                                        "Energy: physics forecast for the planned schedule vs actual"))
    picks = []
    for lab, col in [("NONE", navy), ("JACKET_FOULING", orange)]:
        cand = per_batch[per_batch["fault"] == lab].sort_values("mae_open")
        picks.append((cand.index[len(cand) // 2], lab, col))
    br_ids = bf.loc[bf["y_breached"], "batch_id"]
    if len(br_ids):
        picks.append((br_ids.iloc[0], "BREACH", red))
    for bid, lab, col in picks:
        s = rf[rf["batch_id"] == bid]
        fig.add_trace(go.Scatter(x=s["t_min"], y=s["T_product_c"], name=f"{bid} measured ({lab})", line=dict(color=col, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=s["t_min"], y=s["T_phys_open"], name=f"{bid} physics open", line=dict(color=col, width=1, dash="dot")), 1, 1)
    for lab, col in [("NONE", navy), ("BLADE_WEAR", grey), ("JACKET_FOULING", orange), ("LOW_CW_FLOW", green), ("OVERCHARGE", red), ("MULTIPLE", "#7030A0")]:
        s = per_batch[per_batch["fault"] == lab]
        if len(s):
            fig.add_trace(go.Box(y=s["bias_open"], name=lab, marker_color=col, showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=bt["viscosity_factor"], y=bt["pigment_anchor_kw_resid_pct"], mode="markers", showlegend=False,
                             marker=dict(color=np.where(bt["family_wb"] == 1, navy, orange), size=6, opacity=0.7)), 2, 1)
    fig.add_trace(go.Scatter(x=bf["y_energy_total_kwh"], y=bf["phys_energy_total_kwh"], mode="markers", showlegend=False,
                             marker=dict(color=np.where(bf["y_n_extensions"] == 0, navy, red), size=6, opacity=0.7)), 2, 2)
    lim = [bf["y_energy_total_kwh"].min() * 0.9, bf["y_energy_total_kwh"].max() * 1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=grey, dash="dash"), showlegend=False), 2, 2)
    fig.update_layout(title="Step 4 features - physics baselines and evidence features (SIMULATED data)", template="plotly_white",
                      height=820, font=dict(family="Arial"), legend=dict(orientation="h", y=-0.06))
    fig.update_xaxes(title_text="minutes", row=1, col=1)
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="mean residual, degC", row=1, col=2)
    fig.update_xaxes(title_text="true viscosity lot factor (hidden)", row=2, col=1)
    fig.update_yaxes(title_text="anchor kW residual, %", row=2, col=1)
    fig.update_xaxes(title_text="actual kWh (red = batch had extensions)", row=2, col=2)
    fig.update_yaxes(title_text="physics forecast kWh", row=2, col=2)
    out = ROOT / "reports" / "step4_features_overview.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"  written {out.relative_to(ROOT)}")

    banner("ALL STEP 4 CHECKS PASSED - ready for Step 5 (models)")


if __name__ == "__main__":
    main()
