"""
check_step8.py - validates and tunes the fault-detection rules against the injected faults.

What it does:
  1. runtime baselines (normal plant-vs-physics levels) from the TRAIN batches
  2. runs the twin over every batch and collects the rule statistics + hidden labels
  3. sweeps each rule's threshold -> detection / false-alarm curves; picks operating points
     against the KPI targets (detection >= 80 %, false alarms <= 5 per 100 healthy batches)
  4. scores the tuned rules: per-fault scorecard, confusion matrix (incl. MULTIPLE),
     false flags by recipe / season / lot (does a legitimate recipe effect trip a rule?)
  5. scores the temperature pre-alarm: lead times, watch flags, breach-ETA accuracy
  6. cross-batch drift tracker: CUSUM + trend on the per-batch indicators vs the true
     blade changes and jacket cleanings; remaining-life estimate accuracy
  7. writes the tuned thresholds back into config/assumptions.yaml (values + achieved rates
     in the notes), the rule cards to reports/step8_rule_cards.md, the report HTML, and
     re-runs the Step 7 fleet precompute so the dashboard uses the tuned rules

Run from the project folder:   python scripts/check_step8.py   (needs models/ from Step 5)
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import faults as fl, models as md, runtime as rt  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
RAW, PROC, MODELS, REPORTS, CONFIG = _RAW_DIR, _PROC_DIR, ROOT / "models", ROOT / "reports", ROOT / "config" / "assumptions.yaml"


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main() -> None:
    cfg = load_config()
    REPORTS.mkdir(exist_ok=True)
    tm, bm, card = md.load_models(MODELS)
    ts = pd.read_csv(RAW / "timeseries.csv", parse_dates=["timestamp"])
    batches = pd.read_csv(RAW / "batches.csv", parse_dates=["start_time"]).set_index("batch_id")
    split = pd.read_csv(PROC / "split.csv").set_index("batch_id")
    thr = rt.Thresholds.from_config(cfg)
    min_det, max_false = get(cfg, "kpi_targets.min_fault_detection_rate_pct"), get(cfg, "kpi_targets.max_false_alarms_per_100_batches")

    # ------------------------------------------------------------------- 1
    banner("1. NORMAL LEVELS (train batches): the gap a healthy plant has to the physics")
    tr = split[split["split_random"] == "train"].index
    rows = []
    for bid in tr:
        run = rt.prepare_batch(ts[ts["batch_id"] == bid], cfg, tm, float(batches.loc[bid, "batch_mass_nominal_kg"]),
                               float(batches.loc[bid, "cw_flow_setpoint_m3h"]), thr)
        rows.append(rt.batch_verdict(run))
    Vtr = pd.DataFrame(rows)
    bl = rt.Baselines(float(Vtr["kw_resid_pct_hsd_raw"].median()), float(Vtr["T_resid_replay_hsd_raw"].median()),
                      float(Vtr["u_ratio_hsd_raw"].median()), len(Vtr))
    bl.save(MODELS / "runtime_calibration.json", "medians over the train batches of the SIMULATED dataset")
    print(f"  HSD power {bl.kw_resid_pct:+.1f} % vs physics | product {bl.T_resid_replay_c:+.1f} degC vs measured-power physics | "
          f"implied U {100 * bl.u_ratio:.0f} % of the physics U   ({bl.n_batches} batches)")

    # ------------------------------------------------------------------- 2
    banner("2. RULE STATISTICS FOR EVERY BATCH (what each debounced rule compares with its threshold)")
    t = time.time()
    ind, runs = fl.indicator_table(ts, batches, cfg, tm, thr, bl, verbose=False)
    print(f"  {len(ind)} batches in {time.time() - t:.0f} s")
    print("  how well each statistic tracks the hidden fault state (correlation):")
    for col, tru, text in [("stat_kw_resid", "true_wear", "HSD power vs normal          <-> blade wear"),
                           ("grind_lag_last", "true_wear", "lab grind behind physics      <-> blade wear (corroboration only)"),
                           ("stat_u_drop", "true_fouling", "implied-U drop vs normal     <-> jacket fouling"),
                           ("stat_T_resid", "true_fouling", "temperature vs normal gap    <-> jacket fouling"),
                           ("stat_flow_ratio", "true_flow_frac", "flow / set-point             <-> actual flow fraction"),
                           ("stat_mass_ratio", "true_mass_frac", "mass / nominal               <-> actual mass fraction")]:
        r = ind[[col, tru]].corr().iloc[0, 1]
        print(f"    r = {r:+.2f}   {text}")
    ind.to_csv(REPORTS / "step8_indicators.csv", index=False)

    # ------------------------------------------------------------------- 3
    banner(f"3. THRESHOLD SWEEPS  (targets: detection >= {min_det:.0f} %, false alarms <= {max_false:.0f} per 100 clearly healthy batches)")
    chosen, sweeps = fl.tune_all(ind, cfg)
    for name in ["BLADE_WEAR", "JACKET_FOULING_u", "JACKET_FOULING_T", "LOW_CW_FLOW", "OVERCHARGE"]:
        c, sw = chosen[name], sweeps[name]
        print(f"  {name:18s} stat {c['stat']:16s} {c['direction']:5s}  chosen {c['threshold']:+.3g}  ->  detection {c['detection_pct']:.0f} %, "
              f"false {c['false_pct']:.1f} %   {'MEETS targets' if c['meets_targets'] else 'does NOT meet targets'}")
        pts = sw.iloc[np.linspace(0, len(sw) - 1, 7).astype(int)]
        print("      sweep: " + "  ".join(f"{r['threshold']:+.3g}->{r['detection_pct']:.0f}/{r['false_pct']:.0f}" for _, r in pts.iterrows())
              + "   (threshold -> detection % / false %)")
    fo = chosen["JACKET_FOULING"]
    print("  fouling rule variants:")
    print("    " + fo["variants"].round(1).to_string().replace("\n", "\n    "))
    print(f"  -> fouling uses variant '{fo['mode']}' ({fo['detection_pct']:.0f} % / {fo['false_pct']:.1f} %)")
    ind = fl.apply_thresholds(ind, chosen)

    # ------------------------------------------------------------------- 4
    banner("4. TUNED RULES vs THE INJECTED FAULTS")
    score = fl.per_fault_scorecard(ind)
    print("  per fault (false flags counted on CLEARLY healthy batches; the grey zone is wear 10-18 % / base scale below the label level):")
    print("  " + score.round(1).to_string(index=False).replace("\n", "\n  "))
    before = fl.per_fault_scorecard(ind, prefix="flag_")
    print("  before tuning (Step 7 thresholds): " + ", ".join(f"{r['fault']} {r['detection_pct']:.0f} % / {r['false_per_100']:.1f}" for _, r in before.iterrows()))
    print("\n  confusion: injected class (rows) vs share flagged by each rule (columns, %):")
    print("  " + fl.confusion(ind).to_string().replace("\n", "\n  "))
    for f in fl.FAULTS:
        r = score.set_index("fault").loc[f]
        assert r["detection_pct"] >= min_det, f"{f} detection below target"
        assert r["false_per_100"] <= max_false, f"{f} false alarms above target"
    print("  all four fault rules meet both targets                                   OK")
    # does a legitimate process effect trip a rule?
    ind["viscosity_factor"] = batches.loc[ind["batch_id"], "viscosity_factor"].values
    ind["month"] = pd.to_datetime(ind["start_time"]).dt.month
    ind["mass_t"] = batches.loc[ind["batch_id"], "batch_mass_kg"].values / 1000
    clear_wear, clear_foul = ind["true_wear"] < 0.10, ind["true_fouling"] < 0.0003
    print("\n  false flags on clearly healthy batches by circumstance (should be flat):")
    for label, mask in [("thin lot (viscosity < 0.9)", ind["viscosity_factor"] < 0.9), ("thick lot (viscosity > 1.1)", ind["viscosity_factor"] > 1.1),
                        ("small batch (< 2.5 t)", ind["mass_t"] < 2.5), ("summer (Jun-Sep)", ind["month"].isin([6, 7, 8, 9])),
                        ("winter (Dec-Feb)", ind["month"].isin([12, 1, 2])), ("solvent-based", ind["family"] == "solvent_based")]:
        w = ind[mask & clear_wear]["tuned_BLADE_WEAR"]
        fo_ = ind[mask & clear_foul]["tuned_JACKET_FOULING"]
        print(f"    {label:30s} wear {100 * w.mean():4.0f} % of {len(w):3d}   fouling {100 * fo_.mean():4.0f} % of {len(fo_):3d}")

    # ------------------------------------------------------------------- 5
    banner("5. TEMPERATURE PRE-ALARM")
    pre = fl.prealarm_scorecard(ind, cfg)
    print(f"  breaches {pre['breaches']}, pre-alarmed before the crossing {pre['prealarmed']}; lead time median {pre['lead_median_min']:.0f} min, "
          f"P10 {pre['lead_p10_min']:.0f}, min {pre['lead_min_min']:.0f}  (target >= {get(cfg, 'kpi_targets.breach_lead_time_min')} min: "
          f"{pre['lead_target_met_pct']:.0f} % of breaches)")
    print(f"  watch flags: {pre['watch_flags']} batches pre-alarmed that stayed under the limit - {pre['watch_peak_within_1c']} peaked within 1 degC "
          f"of it, {pre['watch_peak_within_2c']} within 2 degC; median margin {pre['watch_margin_median_c']:.1f} degC")
    print(f"  quiet batches: {pre['quiet_batches']}; the closest any of them came to the limit was {pre['quiet_closest_margin_c']:.1f} degC")
    print(f"  breach ETA at the moment of the pre-alarm: available on {pre['eta_at_prealarm_available']} of {pre['breaches']} breaches, "
          f"median error {pre['eta_error_median_min']:+.1f} min (|error| median {pre['eta_abs_error_median_min']:.1f} min)")
    assert pre["prealarmed"] == pre["breaches"] and pre["lead_min_min"] >= get(cfg, "kpi_targets.breach_lead_time_min")

    # ------------------------------------------------------------------- 6
    banner("6. CROSS-BATCH DRIFT TRACKER (slow wear and base fouling, caught across batches)")
    ind = ind.sort_values("start_time").reset_index(drop=True)
    wear_tr = fl.drift_tracker(ind["kw_resid_median"], 0.0, "down", thr.drift_slack, thr.drift_h, thr.drift_window, chosen["BLADE_WEAR"]["threshold"])
    foul_tr = fl.drift_tracker(ind["u_drop_median"], 0.0, "up", thr.drift_slack, thr.drift_h, thr.drift_window, chosen["JACKET_FOULING_u"]["threshold"])
    ind["wear_cusum"], ind["wear_cusum_alarm"], ind["wear_slope"], ind["wear_batches_to_alarm"] = wear_tr["cusum"], wear_tr["cusum_alarm"], wear_tr["slope_per_batch"], wear_tr["batches_to_alarm"]
    ind["foul_cusum"], ind["foul_cusum_alarm"], ind["foul_slope"], ind["foul_batches_to_alarm"] = foul_tr["cusum"], foul_tr["cusum_alarm"], foul_tr["slope_per_batch"], foul_tr["batches_to_alarm"]
    life = get(cfg, "simulation.faults.blade_life_batches")
    pos = np.arange(len(ind)) % life
    first_label = [int(np.argmax((ind["true_wear"].values[k * life:(k + 1) * life] >= get(cfg, "simulation.faults.blade_wear_label_threshold")))) for k in range(len(ind) // life + 1) if k * life < len(ind)]
    print(f"  blade wear: true blade changes before batches {[int(i) for i in np.where(np.diff(ind['true_wear'].values) < -0.1)[0] + 1]}; "
          f"the label level (18 % wear) is reached at batch ~{first_label[0]} of each {life}-batch life")
    starts = ind["wear_cusum_alarm"] & ~ind["wear_cusum_alarm"].shift(1, fill_value=False)
    for k in range(len(ind) // life + 1):
        seg = ind.iloc[k * life:(k + 1) * life]
        if len(seg) < 20:
            continue
        cus = seg.index[starts.loc[seg.index]].min() if starts.loc[seg.index].any() else None
        per_batch = seg.index[seg["tuned_BLADE_WEAR"]].min() if seg["tuned_BLADE_WEAR"].any() else None
        est_txt = []
        for cp in (60, 80):
            est_rows = seg[(np.arange(len(seg)) == cp)]
            est = est_rows["wear_batches_to_alarm"].iloc[0] if len(est_rows) else NAN
            act = (per_batch - seg.index[0] - cp) if per_batch is not None else NAN
            est_txt.append(f"at batch {cp}: trend says ~{est:.0f}, actual {act:.0f}")
        print(f"    blade life {k + 1}: first per-batch flag at batch {per_batch - seg.index[0] if per_batch is not None else '-'} of life, "
              f"CUSUM trend alarm at {cus - seg.index[0] if cus is not None else '-'}; batches-to-alarm estimate {'; '.join(est_txt)}")
    print("    -> the per-batch rule is the earlier detector; the CUSUM confirms a sustained trend and the remaining-life estimate is rough\n"
          "       (a 30-batch linear trend on a noisy series) - good enough to plan a blade inspection, not to schedule it to the day")
    clean = get(cfg, "simulation.faults.fouling_cleaning_interval_batches")
    print(f"  fouling: jacket cleanings every {clean} batches plus dirty-water episodes; CUSUM alarms at batches "
          f"{[int(i) for i in ind.index[ind['foul_cusum_alarm'] & ~ind['foul_cusum_alarm'].shift(1, fill_value=False)]]}")
    ind.to_csv(REPORTS / "step8_indicators_tuned.csv", index=False)

    # ------------------------------------------------------------------- 7
    banner("7. WRITE BACK: thresholds -> config, rule cards, report, dashboard precompute")
    sc = score.set_index("fault")
    updates = {
        "blade_wear_kw_resid_pct": (round(chosen["BLADE_WEAR"]["threshold"], 2),
                                    f"Tuned Step 8 on simulated data: detection {sc.loc['BLADE_WEAR', 'detection_pct']:.0f} %, false {sc.loc['BLADE_WEAR', 'false_per_100']:.1f}/100 clearly healthy. HSD power below its NORMAL gap to physics by more than this -> BLADE_WEAR"),
        "fouling_u_drop_pct": (round(chosen["JACKET_FOULING_u"]["threshold"], 2),
                               f"Tuned Step 8: U-rule alone detection {chosen['JACKET_FOULING_u']['detection_pct']:.0f} %, false {chosen['JACKET_FOULING_u']['false_pct']:.1f}/100. Implied U below its NORMAL level by more than this"),
        "fouling_T_resid_replay_c": (round(chosen["JACKET_FOULING_T"]["threshold"], 2),
                                     f"Tuned Step 8: T-rule alone detection {chosen['JACKET_FOULING_T']['detection_pct']:.0f} %, false {chosen['JACKET_FOULING_T']['false_pct']:.1f}/100. Product above its NORMAL gap to the measured-power physics by more than this"),
        "fouling_rule_mode": (fo["mode"], f"Chosen by the Step 8 tuner: variant '{fo['mode']}' gives detection {fo['detection_pct']:.0f} %, false {fo['false_pct']:.1f}/100 (u = implied U only, T = temperature residual only, u_or_T, u_and_T)"),
        "low_flow_ratio": (round(chosen["LOW_CW_FLOW"]["threshold"], 2),
                           f"Tuned Step 8: detection {sc.loc['LOW_CW_FLOW', 'detection_pct']:.0f} %, false {sc.loc['LOW_CW_FLOW', 'false_per_100']:.1f}/100. Measured flow / set-point below this -> LOW_CW_FLOW"),
        "overcharge_ratio": (round(max(chosen["OVERCHARGE"]["threshold"], 1.03), 3),
                             f"Tuned Step 8 (floor 1.03 kept above dosing scatter): detection {sc.loc['OVERCHARGE', 'detection_pct']:.0f} %, false {sc.loc['OVERCHARGE', 'false_per_100']:.1f}/100. Load-cell mass / nominal above this -> OVERCHARGE"),
    }
    changed = fl.write_thresholds_to_config(CONFIG, updates)
    cfg2 = load_config()
    thr2 = rt.Thresholds.from_config(cfg2)
    print(f"  config updated: {changed}")
    print(f"  now: wear < {thr2.blade_wear_kw_pct} %, fouling U drop > {thr2.fouling_u_drop_pct} % / T > {thr2.fouling_T_resid_c} degC (mode {thr2.fouling_mode}), "
          f"flow < {thr2.low_flow_ratio}, overcharge > {thr2.overcharge_ratio}")
    cards = fl.rule_cards(chosen, score, pre, thr2, bl, cfg2)
    (REPORTS / "step8_rule_cards.md").write_text(cards)
    print("  reports/step8_rule_cards.md written")

    # ---- report
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        navy, orange, green, grey, red, purple = "#1F3864", "#C55A11", "#548235", "#8A8A8A", "#C00000", "#7030A0"
        fig = make_subplots(rows=3, cols=2, vertical_spacing=0.10, horizontal_spacing=0.08,
                            subplot_titles=("Blade wear: detection and false-alarm rate vs threshold (HSD power vs normal, %)",
                                            "Jacket fouling: the same for the implied-U drop (%) and the temperature residual (degC x10)",
                                            "Per-batch HSD-power residual over the year with the true wear (hidden) and the CUSUM",
                                            "Per-batch implied-U drop with the true fouling (hidden) and the CUSUM",
                                            "Pre-alarm lead time before each breach (min)",
                                            "Watch flags: how close the batch actually came to the limit (degC)"))
        sw = sweeps["BLADE_WEAR"]
        fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["detection_pct"], name="detection %", line=dict(color=navy, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["false_pct"], name="false alarms %", line=dict(color=red, width=2)), 1, 1)
        fig.add_vline(x=chosen["BLADE_WEAR"]["threshold"], line=dict(color=green, dash="dot"), row=1, col=1, annotation_text="chosen")
        sw = sweeps["JACKET_FOULING_u"]
        fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["detection_pct"], name="U-rule detection %", line=dict(color=navy, width=2)), 1, 2)
        fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["false_pct"], name="U-rule false %", line=dict(color=red, width=2)), 1, 2)
        sw = sweeps["JACKET_FOULING_T"]
        fig.add_trace(go.Scatter(x=sw["threshold"] * 10, y=sw["detection_pct"], name="T-rule detection %", line=dict(color=navy, width=1.2, dash="dash")), 1, 2)
        fig.add_trace(go.Scatter(x=sw["threshold"] * 10, y=sw["false_pct"], name="T-rule false %", line=dict(color=red, width=1.2, dash="dash")), 1, 2)
        fig.add_vline(x=chosen["JACKET_FOULING_u"]["threshold"], line=dict(color=green, dash="dot"), row=1, col=2, annotation_text="chosen U")
        x = np.arange(len(ind))
        fig.add_trace(go.Scatter(x=x, y=ind["kw_resid_median"], name="HSD power vs normal (%)", line=dict(color=navy, width=1.2)), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=-20 * ind["true_wear"], name="true wear x(-20) (hidden)", line=dict(color=grey, width=1, dash="dash")), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=-ind["wear_cusum"] / 5, name="CUSUM / -5", line=dict(color=purple, width=1)), 2, 1)
        fig.add_hline(y=chosen["BLADE_WEAR"]["threshold"], line=dict(color=red, dash="dot", width=1), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=ind["u_drop_median"], name="implied-U drop (%)", line=dict(color=orange, width=1.2)), 2, 2)
        fig.add_trace(go.Scatter(x=x, y=ind["true_fouling"] * 15000, name="true fouling x15000 (hidden)", line=dict(color=grey, width=1, dash="dash")), 2, 2)
        fig.add_trace(go.Scatter(x=x, y=ind["foul_cusum"] / 5, name="CUSUM / 5", line=dict(color=purple, width=1)), 2, 2)
        fig.add_hline(y=chosen["JACKET_FOULING_u"]["threshold"], line=dict(color=red, dash="dot", width=1), row=2, col=2)
        br = ind[ind["breached"]]
        fig.add_trace(go.Bar(x=br["batch_id"], y=br["t_breach_min"] - br["t_prealarm_min"], marker_color=navy, showlegend=False), 3, 1)
        fig.add_hline(y=get(cfg, "kpi_targets.breach_lead_time_min"), line=dict(color=red, dash="dot"), row=3, col=1, annotation_text="target")
        watch = ind[ind["flag_TEMP_PREALARM"] & ~ind["breached"]]
        fig.add_trace(go.Histogram(x=watch["limit_c"] - watch["peak_T"], nbinsx=12, marker_color=orange, showlegend=False), 3, 2)
        fig.update_layout(title="Step 8 - fault detection validated against the injected faults (SIMULATED data)", template="plotly_white",
                          height=1150, font=dict(family="Arial"), legend=dict(orientation="h", y=-0.04))
        fig.update_xaxes(title_text="threshold (%)", row=1, col=1); fig.update_yaxes(title_text="%", row=1, col=1)
        fig.update_xaxes(title_text="threshold (U %, T degC x10)", row=1, col=2); fig.update_yaxes(title_text="%", row=1, col=2)
        fig.update_xaxes(title_text="batch number", row=2, col=1); fig.update_xaxes(title_text="batch number", row=2, col=2)
        fig.update_yaxes(title_text="min", row=3, col=1); fig.update_xaxes(title_text="margin to limit at the peak, degC", row=3, col=2)
        out = REPORTS / "step8_fault_detection.html"
        fig.write_html(out, include_plotlyjs=True)
        print(f"  {out.relative_to(ROOT)} written")
    except ImportError:
        print("  plotly not installed - report skipped")

    # ---- dashboard precompute with the tuned thresholds
    import check_step7
    print("  re-running the Step 7 fleet precompute with the tuned thresholds...")
    check_step7.main(screenshots=False)
    banner("ALL STEP 8 CHECKS PASSED - every fault rule meets its targets; config updated; ready for Step 9 (path to live data)")


NAN = float("nan")

if __name__ == "__main__":
    main()
