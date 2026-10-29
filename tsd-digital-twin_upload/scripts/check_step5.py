"""
check_step5.py - trains the twin's models and reports the results honestly.

What it does:
  1. trains on the random-split TRAIN batches, scores on the TEST batches (never seen)
  2. temperature: physics vs ML-only vs hybrid forecast / nowcast / online, with pass rates,
     by family and by injected fault, and the worst batches named
  3. energy and time to target Hegman: physics vs ridge vs gradient boosting; energy given the
     schedule, energy forecast (one-stage and two-stage), time vs lab and vs the true crossing
  4. peak temperature and limit breach, with hits / misses / false alarms
  5. robustness: retrain on the first 70 % of the year, score on the last 30 % (new seasons,
     second blade life, second fouling cycle)
  6. permutation importance on the test batches for the four GB models
  7. saves models/ and writes reports/step5_model_report.html

Run from the project folder:   python scripts/check_step5.py   (needs data/processed from Step 4)
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

from tsd_twin import features as ft, models as md  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
PROC, RAW, MODELS = _PROC_DIR, _RAW_DIR, ROOT / "models"


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def split_sets(split: pd.DataFrame, col: str) -> tuple[set, set]:
    return (set(split.loc[split[col] == "train", "batch_id"]), set(split.loc[split[col] == "test", "batch_id"]))


def train_and_score(rf, bf, tr, te, cfg, truth, label: str):
    r_tr, r_te = rf[rf["batch_id"].isin(tr)], rf[rf["batch_id"].isin(te)]
    b_tr, b_te = bf[bf["batch_id"].isin(tr)], bf[bf["batch_id"].isin(te)]
    t = time.time()
    tm = md.TemperatureModels().fit(r_tr)
    bm = md.BatchModels().fit(b_tr)
    print(f"  [{label}] trained on {len(tr)} batches ({len(r_tr):,} rows) in {time.time() - t:.0f} s; scoring {len(te)} unseen batches")
    pred_T = tm.predict(r_te)
    r_te = r_te.copy()
    r_te["fault"] = r_te["batch_id"].map(truth["fault_injected"])
    r_te["family"] = np.where(r_te["family_wb"] == 1, "WB", "SB")
    per_batch, fleet = md.temperature_scorecard(r_te, pred_T, cfg, label_col="fault")
    per_batch["family"] = per_batch["batch_id"].map(r_te.groupby("batch_id")["family"].first())
    pred_B = bm.predict(b_te, cfg)
    score_B = md.batch_scorecard(b_te, pred_B, cfg, truth_time=truth["time_to_target_hegman_true_min"])
    return dict(tm=tm, bm=bm, r_te=r_te, b_te=b_te, pred_T=pred_T, per_batch=per_batch, fleet=fleet, pred_B=pred_B, score_B=score_B)


def main() -> None:
    cfg = load_config()
    (ROOT / "reports").mkdir(exist_ok=True)
    if not (PROC / "row_features.csv.gz").exists():
        raise SystemExit("data/processed not found - run scripts/check_step4.py first")
    d = ft.load_processed(PROC)
    rf, bf, split = d["row_features"], d["batch_features"], d["split"]
    truth = pd.read_csv(RAW / "batches.csv").set_index("batch_id")          # labels for REPORTING only
    tgt_mae, tgt_max = get(cfg, "kpi_targets.temp_mae_c"), get(cfg, "kpi_targets.temp_max_abs_error_c")
    tgt_e, tgt_t = get(cfg, "kpi_targets.energy_error_pct"), get(cfg, "kpi_targets.dispersion_time_error_pct")
    pass_req = get(cfg, "kpi_targets.pass_rate_pct")

    # ------------------------------------------------------------------- 1
    banner("1. TRAIN (random split, stratified by family, BY BATCH)")
    tr, te = split_sets(split, "split_random")
    R = train_and_score(rf, bf, tr, te, cfg, truth, "random split")

    # ------------------------------------------------------------------- 2
    banner(f"2. TEMPERATURE OVER THE BATCH  (per-batch MAE; target MAE <= {tgt_mae} degC, max error <= {tgt_max} degC, "
           f"pass rate >= {pass_req} %)")
    fleet = R["fleet"].copy()
    print(fleet.round(2).to_string(index=False))
    print("\n  what each predictor knows:")
    print("    physics          Step 2 with the set-points and utilities only")
    print("    ml_only          gradient boosting on the measured tags, no physics (the foil)")
    print("    hybrid_forecast  physics + learned residual from recipe/set-points/utilities  -> used by WHAT-IF")
    print("    hybrid_nowcast   + measured kW, jacket outlet, cooling duty                   -> live display")
    print("    online           + the batch's own last 5 minutes                            -> best nowcast with history")
    print("    ahead            forecast inputs + the last 5 minutes, nothing measured at the target time -> TRUE 5-min-ahead, for alerts")
    pb = R["per_batch"]
    print("\n  median MAE by family and by injected fault (truth labels, reporting only):")
    cols = [f"{m}_mae" for m in md.T_MODEL_NAMES]
    print(pb.groupby("family")[cols].median().round(2).rename(columns=lambda c: c.replace("_mae", "")).to_string())
    print(pb.groupby("fault")[cols].median().round(2).rename(columns=lambda c: c.replace("_mae", "")).to_string())
    worst = pb.nlargest(3, "hybrid_forecast_mae")[["batch_id", "fault", "family", "physics_mae", "hybrid_forecast_mae", "hybrid_forecast_max", "hybrid_forecast_peak_err"]]
    print("\n  three worst test batches for the hybrid forecast:")
    print(worst.round(2).to_string(index=False))
    hf = fleet.set_index("model").loc["hybrid_forecast"]
    hp = fleet.set_index("model").loc["physics"]
    print(f"\n  VERDICT temperature: hybrid forecast median MAE {hf['mae_median']:.2f} degC, P90 {hf['mae_p90']:.2f}, "
          f"pass {hf[f'pass_mae<={tgt_mae}']:.0f} % (physics alone {hp[f'pass_mae<={tgt_mae}']:.0f} %); "
          f"peak-temperature bias {hf['peak_bias']:+.2f} degC (physics {hp['peak_bias']:+.2f}).")
    assert hf[f"pass_mae<={tgt_mae}"] >= pass_req, "hybrid temperature model misses the pass-rate target"

    # ------------------------------------------------------------------- 3
    banner("3. ENERGY PER BATCH AND TIME TO TARGET HEGMAN  (batch level, known at HSD start + 5 min)")
    sb = R["score_B"].set_index("model")
    print(f"  3a. ENERGY GIVEN THE SCHEDULE (what-if question; target +/-{tgt_e} %):")
    print(sb[["esched_mape", "esched_bias", "esched_p90", f"esched_pass<={tgt_e}%"]].round(2).to_string())
    print(f"\n  3b. ENERGY FORECAST BEFORE THE LAB OUTCOME IS KNOWN (pre-batch question):")
    print(sb[["efc_mape", "efc_bias", f"efc_pass<={tgt_e}%", "efc_mape_no_ext", "efc_mape_ext"]].round(2).to_string())
    print("      one-stage: a single model hedges between 'lab passes' and 'lab fails' and is wrong for both")
    print(sb.loc[["linear", "gb"], ["efc2_mape", "efc2_bias", f"efc2_pass<={tgt_e}%", "efc2_schedule_right_%", "efc2_mape_sched_right"]].round(2).to_string())
    print("      two-stage: predict which lab check passes -> schedule -> energy for that schedule")
    print(f"\n  3c. TIME TO TARGET HEGMAN (target +/-{tgt_t} %):")
    print(sb[["time_n", "time_mae_min", "time_bias_min", "time_mape", f"time_pass<={tgt_t}%", "time_within_one_check",
              "time_unpredicted", "time_true_mae_min", f"time_true_pass<={tgt_t}%"]].round(2).to_string())
    first, every = get(cfg, "data.lab_hegman_first_check_min"), get(cfg, "data.lab_hegman_interval_min")
    print(f"      the lab only looks at {first} min and every {every} min after: the 'actual' time to target is quantised,")
    print(f"      so +/-{tgt_t} % of a 35-min grind (3.5 min) is finer than the measurement itself.")
    es, ef2, tt = sb.loc["gb"], sb.loc["linear"], sb.loc["linear"]
    print(f"\n  VERDICT energy: given the schedule, {es['esched_mape']:.2f} % MAPE, pass {es[f'esched_pass<={tgt_e}%']:.0f} %  - MET.")
    print(f"                  forecast before the lab outcome: two-stage {ef2['efc2_mape']:.1f} % MAPE, pass {ef2[f'efc2_pass<={tgt_e}%']:.0f} % "
          f"- NOT MET; the miss is the lab-check lottery ({100 - ef2['efc2_schedule_right_%']:.0f} % of schedules called wrong), "
          f"not the energy model ({ef2['efc2_mape_sched_right']:.2f} % when the schedule is right).")
    print(f"  VERDICT time:   MAE {tt['time_mae_min']:.1f} min vs lab, {tt['time_true_mae_min']:.1f} min vs the true crossing; "
          f"within one lab check {tt['time_within_one_check']:.0f} %; +/-{tgt_t} % pass {tt[f'time_pass<={tgt_t}%']:.0f} % - NOT MET as measured by the lab.")
    print("                  Proposed replacement target: predict the passing lab check (within one check) >= 90 %, or MAE <= 5 min")
    print("                  against an inline grind measurement if one is installed (Step 9).")

    # ------------------------------------------------------------------- 4
    banner("4. PEAK TEMPERATURE AND LIMIT BREACH  (batch level, at HSD start + 5 min)")
    print(sb[["peak_mae", "peak_bias", "breach_actual", "breach_hits", "breach_misses", "breach_false", "prealarm_catches", "prealarm_false"]].to_string())
    print(f"  only {int(sb.loc['gb', 'breach_actual'])} of the {len(R['b_te'])} test batches breach - counts, not percentages, at this sample size.")
    print("  the pre-alarm (peak within 3 degC of the limit) is the useful flag: it catches the breaches at the cost of some")
    print("  'watch' flags on batches that end up just under the limit. Row-level 5-min-ahead warnings come in Step 8.")

    # ------------------------------------------------------------------- 5
    banner("5. ROBUSTNESS: TRAIN ON THE FIRST 70 % OF THE YEAR, TEST ON THE LAST 30 %")
    tr2, te2 = split_sets(split, "split_time")
    R2 = train_and_score(rf, bf, tr2, te2, cfg, truth, "time split")
    f2 = R2["fleet"].set_index("model")
    print("  temperature (per-batch MAE):")
    print(f2[["mae_median", "mae_p90", f"pass_mae<={tgt_mae}", "peak_bias"]].round(2).to_string())
    s2 = R2["score_B"].set_index("model")
    print("  batch models:")
    print(s2[["esched_mape", "efc2_mape", f"efc2_pass<={tgt_e}%", "time_mae_min", "time_within_one_check", "peak_mae",
              "breach_actual", "breach_hits", "breach_misses", "breach_false", "prealarm_catches", "prealarm_false"]].round(2).to_string())
    print(f"  -> test period {split.loc[split['split_time'] == 'test', 'start_time'].min():%b %Y} .. "
          f"{split['start_time'].max():%b %Y}: the model had seen summer (Jun-Aug) in training but not Sep-Oct, nor the second blade life.")

    # ------------------------------------------------------------------- 6
    banner("6. WHAT THE MODELS USE  (permutation importance on the test batches, MAE increase when shuffled)")
    r_te, b_te, tm, bm = R["r_te"], R["b_te"], R["tm"], R["bm"]
    sub = r_te.sample(min(20000, len(r_te)), random_state=1)
    imp_T = md.importance(tm.forecast, sub[md.FEATURES_T_FORECAST], sub["resid_open_c"], n_repeats=5)
    Xb, Xs = bm._X(b_te, bm.features), bm._X(b_te, bm.features_sched)
    imp_E = md.importance(bm.energy_sched_gb, Xs, b_te["y_energy_total_kwh"] / b_te["phys_energy_actual_sched_kwh"])
    imp_t = md.importance(bm.time_gb, Xb, b_te["y_time_to_target_lab_min"])
    imp_P = md.importance(bm.peak_gb, Xb, b_te["y_peak_temp_c"] - b_te["phys_peak_temp_c"])
    for title, imp, unit in [("temperature residual (hybrid forecast)", imp_T, "degC"),
                             ("energy ratio given schedule", imp_E, "ratio"),
                             ("time to target Hegman", imp_t, "min"),
                             ("peak-temperature offset", imp_P, "degC")]:
        top = imp.head(6)
        print(f"  {title}:")
        for _, r in top.iterrows():
            print(f"     {r['feature']:32s} +{r['importance']:.3f} {unit}")
    imps = {"temperature_residual": imp_T, "energy_given_schedule": imp_E, "time_to_target": imp_t, "peak_offset": imp_P}
    for k, v in imps.items():
        v.to_csv(ROOT / "reports" / f"step5_importance_{k}.csv", index=False)

    # ------------------------------------------------------------------- 7
    banner("7. SAVE MODELS AND REPORT")
    meta = {"trained_on": "random split train batches (SIMULATED data)", "n_train_batches": len(tr), "n_test_batches": len(te),
            "config_version": get(cfg, "meta.version"), "temperature_fleet": R["fleet"].round(3).to_dict("records"),
            "batch_scorecard": R["score_B"].round(3).to_dict("records"),
            "time_split_temperature": R2["fleet"].round(3).to_dict("records"), "time_split_batch": R2["score_B"].round(3).to_dict("records"),
            "note": "No result here is plant performance. Retrain on real batches (Step 9) before relying on any number."}
    md.save_models(tm, bm, MODELS, meta)
    R["fleet"].round(3).to_csv(ROOT / "reports" / "step5_temperature_scorecard.csv", index=False)
    R["score_B"].round(3).to_csv(ROOT / "reports" / "step5_batch_scorecard.csv", index=False)
    pb.round(3).to_csv(ROOT / "reports" / "step5_temperature_per_batch.csv", index=False)
    print(f"  models/temperature_models.joblib, models/batch_models.joblib, models/model_card.json; scorecards in reports/")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed - report skipped, everything else passed.")
        banner("STEP 5 DONE - see the verdicts above; ready for Step 6 (what-if)")
        return
    navy, orange, grey, green, red, purple = "#1F3864", "#C55A11", "#8A8A8A", "#548235", "#C00000", "#7030A0"
    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.10, horizontal_spacing=0.08,
                        subplot_titles=("Test batch: measured vs physics vs hybrid forecast vs online (5 min ahead)",
                                        "Per-batch temperature MAE on unseen batches, by predictor",
                                        "Energy for the schedule actually run: GB vs actual",
                                        "Energy forecast before the lab outcome: two-stage vs actual (red = schedule called wrong)",
                                        "Time to target Hegman: ridge prediction vs lab reading (grey band = one lab check)",
                                        "Top features of the temperature residual model (permutation importance)"))
    br = R["b_te"].loc[R["b_te"]["y_breached"], "batch_id"]
    bid = br.iloc[0] if len(br) else pb.sort_values("physics_mae").iloc[-1]["batch_id"]
    s = r_te[r_te["batch_id"] == bid]
    p = R["pred_T"].loc[s.index]
    fig.add_trace(go.Scatter(x=s["t_min"], y=s["T_product_c"], name=f"{bid} measured", line=dict(color="black", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=s["t_min"], y=p["T_physics"], name="physics", line=dict(color=grey, width=1.5, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=s["t_min"], y=p["T_hybrid_forecast"], name="hybrid forecast", line=dict(color=navy, width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=s["t_min"], y=p["T_online"], name="online 5-min-ahead", line=dict(color=orange, width=1.5, dash="dash")), 1, 1)
    for name, col in zip(md.T_MODEL_NAMES, [grey, purple, navy, green, orange, red]):
        fig.add_trace(go.Box(y=pb[f"{name}_mae"], name=name, marker_color=col, showlegend=False), 1, 2)
    fig.add_hline(y=tgt_mae, line=dict(color=red, dash="dot"), row=1, col=2, annotation_text=f"target {tgt_mae} degC")
    m = R["b_te"].merge(R["pred_B"], on="batch_id")
    lim = [m["y_energy_total_kwh"].min() * 0.9, m["y_energy_total_kwh"].max() * 1.05]
    fig.add_trace(go.Scatter(x=m["y_energy_total_kwh"], y=m["esched_gb"], mode="markers", showlegend=False, marker=dict(color=navy, size=6)), 2, 1)
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=grey, dash="dash"), showlegend=False), 2, 1)
    right = m["n_ext_pred_linear"] == m["y_n_extensions"]
    fig.add_trace(go.Scatter(x=m["y_energy_total_kwh"], y=m["efc2_linear"], mode="markers", showlegend=False,
                             marker=dict(color=np.where(right, navy, red), size=6)), 2, 2)
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=grey, dash="dash"), showlegend=False), 2, 2)
    ok = m["y_time_to_target_lab_min"].notna()
    tl = [10, 70]
    fig.add_trace(go.Scatter(x=tl, y=[tl[0] + every, tl[1] + every], mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=tl, y=[tl[0] - every, tl[1] - every], mode="lines", line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                             fillcolor="rgba(140,140,140,0.15)", showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=m.loc[ok, "y_time_to_target_lab_min"], y=m.loc[ok, "time_linear"], mode="markers", showlegend=False,
                             marker=dict(color=np.where(m.loc[ok, "family_wb"] == 1, navy, orange), size=6)), 3, 1)
    fig.add_trace(go.Scatter(x=tl, y=tl, mode="lines", line=dict(color=grey, dash="dash"), showlegend=False), 3, 1)
    top = imp_T.head(10).iloc[::-1]
    fig.add_trace(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color=navy, showlegend=False), 3, 2)
    fig.update_layout(title="Step 5 models - results on unseen SIMULATED batches", template="plotly_white", height=1150,
                      font=dict(family="Arial"), legend=dict(orientation="h", y=-0.03))
    fig.update_xaxes(title_text="minutes", row=1, col=1); fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="MAE, degC", row=1, col=2)
    fig.update_xaxes(title_text="actual kWh", row=2, col=1); fig.update_yaxes(title_text="predicted kWh", row=2, col=1)
    fig.update_xaxes(title_text="actual kWh", row=2, col=2); fig.update_yaxes(title_text="forecast kWh", row=2, col=2)
    fig.update_xaxes(title_text="lab time to target, min", row=3, col=1); fig.update_yaxes(title_text="predicted, min", row=3, col=1)
    fig.update_xaxes(title_text="MAE increase when shuffled, degC", row=3, col=2)
    out = ROOT / "reports" / "step5_model_report.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"  written {out.relative_to(ROOT)}")

    banner("STEP 5 DONE - temperature target MET; energy MET given the schedule, NOT MET as a pre-lab forecast;\n"
           "dispersion time NOT MET at +/-10 % (lab quantisation) - see the verdicts above. Ready for Step 6 (what-if)")


if __name__ == "__main__":
    main()
