"""
check_step6.py - calibrates, validates and demonstrates the what-if simulator and the optimiser.

What it does:
  1. calibrates the two scalar corrections from the TRAIN batches -> models/whatif_calibration.json
  2. validates the what-if against the TEST batches: feed each batch's recipe, set-points and
     utilities as a scenario, compare the predicted kWh, peak temperature, dispersion time and
     breach flag with what the (simulated) plant actually did
  3. the four what-if questions every plant engineer asks, answered with numbers
  4. sweeps: peak temperature and kWh/t vs rpm, batch size, chilled-water inlet
  5. optimiser on the reference matt (winter and summer) and the gloss enamel; proves the result
     passes every check and that its neighbours are worse or infeasible
  6. plot -> reports/step6_whatif.html

Run from the project folder:   python scripts/check_step6.py   (needs models/ from Step 5)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = Path(os.environ.get("TSD_DATA_DIR", ROOT / "data" / "raw"))
_PROC_DIR = ROOT / "data" / ("processed" if _RAW_DIR.name == "raw" else f"processed_{_RAW_DIR.name}")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import whatif as wi  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
PROC, RAW, MODELS = _PROC_DIR, _RAW_DIR, ROOT / "models"


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main() -> None:
    cfg = load_config()
    (ROOT / "reports").mkdir(exist_ok=True)
    if not (MODELS / "temperature_models.joblib").exists():
        raise SystemExit("models/ not found - run scripts/check_step5.py first")
    bf = pd.read_csv(PROC / "batch_features.csv")
    split = pd.read_csv(PROC / "split.csv")
    truth = pd.read_csv(RAW / "batches.csv").set_index("batch_id")
    tr_ids = set(split.loc[split["split_random"] == "train", "batch_id"])
    te_ids = set(split.loc[split["split_random"] == "test", "batch_id"])

    # ------------------------------------------------------------------- 1
    banner("1. CALIBRATION FROM THE TRAIN BATCHES (two numbers, both to be refitted on real batches first)")
    tr = bf[bf["batch_id"].isin(tr_ids)]
    energy_factor = float((tr["y_energy_total_kwh"] / tr["phys_energy_actual_sched_kwh"]).median())
    every = get(cfg, "data.lab_hegman_interval_min")
    ok = tr["y_time_to_target_lab_min"].notna() & tr["phys_time_to_target_min"].notna()
    grind_factor = float(((tr.loc[ok, "y_time_to_target_lab_min"] - every / 2) / tr.loc[ok, "phys_time_to_target_min"]).median())
    wi.save_calibration(MODELS / "whatif_calibration.json", energy_factor, grind_factor, len(tr),
                        "median measured/physics ratios on the train batches of the SIMULATED dataset")
    twin = wi.load_twin(MODELS)
    print(f"  energy factor      {energy_factor:.4f}  (measured kWh / physics kWh for the schedule actually run)")
    print(f"  grind-time factor  {grind_factor:.4f}  (lab time to target, de-biased by half a check interval, / physics time)")
    print(f"  residual model: {twin.source}")

    # ------------------------------------------------------------------- 2
    banner("2. VALIDATION: the what-if replays each TEST batch's settings and is compared with what happened")
    rows = []
    t = time.time()
    for _, b in bf[bf["batch_id"].isin(te_ids)].iterrows():
        s = wi.Settings(b["product_family"], b["batch_mass_kg"], b["solids_pct"], b["pvc_pct"], b["target_hegman"],   # batch_mass_kg here = recipe nominal (Step 9)
                        b["hsd_rpm_setpoint"], b["anchor_rpm_hsd"], b["planned_dispersion_min"], b["T_cw_in_at_t0_c"],
                        b["cw_flow_m3h_at_t0"], b["T_start_c"], b["T_ambient_c"], recipe_id=b["recipe_id"])
        r = wi.run_scenario(s, cfg, twin)
        sm = r.summary
        rows.append({"batch_id": b["batch_id"], "family": b["product_family"],
                     "kwh_pred": sm["energy_total_kwh"], "kwh_actual": b["y_energy_total_kwh"],
                     "peak_pred": sm["peak_temp_c"], "peak_actual": b["y_peak_temp_c"],
                     "disp_pred": sm["dispersion_min"], "disp_actual": b["y_dispersion_min"],
                     "ext_pred": sm["n_extensions"], "ext_actual": b["y_n_extensions"],
                     "breach_pred": sm["peak_temp_c"] > sm["temp_limit_c"], "breach_actual": bool(b["y_breached"]),
                     "prealarm_pred": sm["peak_temp_c"] > sm["temp_limit_c"] - 3, "fault": truth.loc[b["batch_id"], "fault_injected"]})
    v = pd.DataFrame(rows)
    print(f"  {len(v)} test batches replayed in {time.time() - t:.1f} s ({1000 * (time.time() - t) / len(v):.0f} ms each)")
    e_err = 100 * (v["kwh_pred"] - v["kwh_actual"]) / v["kwh_actual"]
    sched_right = v["ext_pred"] == v["ext_actual"]
    pk_err = v["peak_pred"] - v["peak_actual"]
    tgt_e = get(cfg, "kpi_targets.energy_error_pct")
    print(f"  energy:   MAPE {e_err.abs().mean():.1f} %, bias {e_err.mean():+.1f} %, within +/-{tgt_e} %: {100 * (e_err.abs() <= tgt_e).mean():.0f} % "
          f"| schedule (extensions) called right {100 * sched_right.mean():.0f} %; MAPE when right {e_err[sched_right].abs().mean():.1f} %")
    print(f"  peak T:   MAE {pk_err.abs().mean():.2f} degC, bias {pk_err.mean():+.2f} degC, P90 {pk_err.abs().quantile(0.9):.2f}")
    print(f"  breach:   actual {int(v['breach_actual'].sum())}, predicted {int(v['breach_pred'].sum())}, "
          f"hits {int((v['breach_actual'] & v['breach_pred']).sum())}, false {int((~v['breach_actual'] & v['breach_pred']).sum())}; "
          f"pre-alarm catches {int((v['breach_actual'] & v['prealarm_pred']).sum())}, false pre-alarms {int((~v['breach_actual'] & v['prealarm_pred']).sum())}")
    print("  by fault (what-if assumes a healthy machine, so faulted batches are where it is wrong):")
    v["abs_e"], v["abs_pk"] = e_err.abs(), pk_err.abs()
    print(v.groupby("fault").agg(n=("batch_id", "size"), energy_mape=("abs_e", "mean"), peak_mae=("abs_pk", "mean"),
                                 peak_bias=("peak_pred", lambda s: (s - v.loc[s.index, "peak_actual"]).mean())).round(2).to_string())
    assert pk_err.abs().mean() < 1.5, "what-if peak temperature too far from the measured peaks"
    v.to_csv(ROOT / "reports" / "step6_whatif_validation.csv", index=False)

    # ------------------------------------------------------------------- 3
    banner("3. FOUR QUESTIONS A PLANT ENGINEER ASKS  (reference matt, 3,000 kg, target Hegman 6)")
    base = wi.Settings("water_based", 3000, 58, 55, 6.0, hsd_rpm=950, anchor_rpm=35, dispersion_min=35,
                       cw_inlet_c=22, cw_flow_m3h=10, T_start_c=30, T_ambient_c=32)
    q = {
        "base: 950 rpm, 35 min, CW 22 degC @ 10 m3/h": base,
        "Q1 summer: CW 28 degC, raw material 33 degC": replace(base, cw_inlet_c=28, T_start_c=33, T_ambient_c=35),
        "Q2 summer + 1,050 rpm to finish faster": replace(base, cw_inlet_c=28, T_start_c=33, T_ambient_c=35, hsd_rpm=1050),
        "Q3 summer + 4,400 kg putty-size batch": replace(base, cw_inlet_c=28, T_start_c=33, T_ambient_c=35, batch_mass_kg=4400, solids_pct=68, pvc_pct=75, target_hegman=4.5),
        "Q4 base but stop at the first passing lab check": replace(base, stop_at_first_pass=True),
    }
    res = {k: wi.run_scenario(s, cfg, twin) for k, s in q.items()}
    cmp = wi.compare_scenarios(res)
    print(cmp.loc[["verdict", "energy_total_kwh", "specific_energy_kwh_per_t", "peak_temp_c", "margin_to_limit_c", "breach_at_min",
                   "time_to_target_hegman_min", "dispersion_min", "n_extensions", "batch_cycle_min", "throughput_t_per_h",
                   "hsd_load_peak_pct", "fails", "warnings"]].to_string())
    assert res["Q2 summer + 1,050 rpm to finish faster"].summary["verdict"] == "FAIL"
    assert res["base: 950 rpm, 35 min, CW 22 degC @ 10 m3/h"].summary["verdict"] == "PASS"

    # ------------------------------------------------------------------- 4
    banner("4. SWEEPS  (same reference matt; peak temperature and kWh/t)")
    sweeps = {}
    for label, key, values, extra in [
        ("HSD rpm (planned 35 min)", "hsd_rpm", [775, 825, 875, 925, 975, 1025, 1061], {}),
        ("batch mass kg (summer)", "batch_mass_kg", [2000, 2500, 3000, 3500, 4000, 4400], dict(cw_inlet_c=28, T_start_c=33, T_ambient_c=35)),
        ("CW inlet degC (10 m3/h)", "cw_inlet_c", [12, 16, 20, 24, 28], {}),
        ("CW flow m3/h (summer)", "cw_flow_m3h", [5, 7.5, 10, 12.5, 15], dict(cw_inlet_c=28, T_start_c=33, T_ambient_c=35)),
    ]:
        pts = []
        for x in values:
            r = wi.run_scenario(replace(base, **{key: x}, **extra), cfg, twin).summary
            pts.append({key: x, "peak_c": r["peak_temp_c"], "kwh_t": r["specific_energy_kwh_per_t"], "t_target": r["time_to_target_hegman_min"],
                        "disp": r["dispersion_min"], "ext": r["n_extensions"], "verdict": r["verdict"]})
        sweeps[label] = pd.DataFrame(pts)
        print(f"  {label}:")
        print("   " + sweeps[label].to_string(index=False).replace("\n", "\n   "))
    sw = sweeps["CW flow m3/h (summer)"]
    print(f"  -> chilled-water flow 5 -> 15 m3/h moves the summer peak by only {sw['peak_c'].iloc[0] - sw['peak_c'].iloc[-1]:.1f} degC: "
          f"the product-side film, not the jacket, limits the cooling (h_product ~350 vs h_jacket ~2,500 W/m2K).")

    # ------------------------------------------------------------------- 5
    banner("5. OPTIMISER: lowest kWh/t under the temperature limit")
    cases = {"matt, winter (CW 16 degC)": replace(base, cw_inlet_c=16, T_start_c=26, T_ambient_c=24),
             "matt, summer (CW 28 degC)": replace(base, cw_inlet_c=28, T_start_c=33, T_ambient_c=35),
             "gloss enamel, summer": wi.Settings("solvent_based", 3000, 65, 18, 7.0, 912, 35, 40, 28, 10, 33, 35)}
    opts = {}
    for name, s0 in cases.items():
        t = time.time()
        opt = wi.optimise(s0, cfg, twin)
        opts[name] = opt
        base_r = wi.run_scenario(s0, cfg, twin).summary
        print(f"\n  {name}  [{opt['n_evaluated']} candidates, {opt['n_feasible']} feasible, {time.time() - t:.0f} s]")
        print(f"    recipe as given : {base_r['specific_energy_kwh_per_t']:.2f} kWh/t, peak {base_r['peak_temp_c']:.1f} degC, "
              f"dispersion {base_r['dispersion_min']:.0f} min, {base_r['verdict']}")
        print("    " + opt["explanation"].replace(". ", ".\n    "))
        best = opt["best"]
        assert best.summary["verdict"] == "PASS", "optimiser returned a failing scenario"
        assert best.summary["n_extensions"] == 0
        assert best.summary["peak_temp_c"] <= best.summary["temp_limit_c"] - 1.0 + 1e-6
        c = opt["candidates"]
        cheaper = c[(c["kwh_per_t"] < opt["best_row"]["kwh_per_t"] - 0.005) & c["feasible"]]   # ties within 0.01 kWh/t go to the cooler point
        assert cheaper.empty, "a cheaper feasible candidate exists"
        print(f"    proof: passes every check, no extensions, peak {best.summary['peak_temp_c']} <= {best.summary['temp_limit_c'] - 1.0}; "
              f"no cheaper feasible candidate among {len(c)} evaluated")
        print(f"    top rejected reasons: {c.loc[~c['feasible'], 'reason'].str.split(';').str[0].value_counts().head(3).to_dict()}")

    # ------------------------------------------------------------------- 6
    banner("6. PLOT")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed - plot skipped.")
        banner("ALL STEP 6 CHECKS PASSED - ready for Step 7 (dashboard)")
        return
    navy, orange, green, purple, red, grey = "#1F3864", "#C55A11", "#548235", "#7030A0", "#C00000", "#8A8A8A"
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=("Scenario comparison: predicted product temperature",
                                        "What-if validation on test batches: predicted vs actual peak temperature",
                                        "Sweep: peak temperature and kWh/t vs HSD rpm (summer)",
                                        "Optimiser candidates, summer matt: kWh/t vs peak temperature"))
    for (name, r), col in zip(res.items(), [navy, orange, red, purple, green]):
        fig.add_trace(go.Scatter(x=r.curve["t_min"], y=r.curve["T_pred_c"], name=name[:40], line=dict(color=col, width=2)), 1, 1)
    fig.add_hline(y=45, line=dict(color=red, dash="dot"), row=1, col=1, annotation_text="45 degC limit")
    fig.add_trace(go.Scatter(x=v["peak_actual"], y=v["peak_pred"], mode="markers", showlegend=False,
                             marker=dict(color=np.where(v["family"] == "water_based", navy, orange), size=6)), 1, 2)
    lim = [v["peak_actual"].min() - 1, v["peak_actual"].max() + 1]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=grey, dash="dash"), showlegend=False), 1, 2)
    pts = []
    for rpm in range(775, 1062, 25):
        r = wi.run_scenario(replace(base, hsd_rpm=rpm, cw_inlet_c=28, T_start_c=33, T_ambient_c=35), cfg, twin).summary
        pts.append((rpm, r["peak_temp_c"], r["specific_energy_kwh_per_t"]))
    pts = pd.DataFrame(pts, columns=["rpm", "peak", "kwh_t"])
    fig.add_trace(go.Scatter(x=pts["rpm"], y=pts["peak"], name="peak degC", line=dict(color=red, width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=pts["rpm"], y=pts["kwh_t"] * 3, name="kWh/t x3", line=dict(color=navy, width=2, dash="dash")), 2, 1)
    fig.add_hline(y=45, line=dict(color=red, dash="dot"), row=2, col=1)
    c = opts["matt, summer (CW 28 degC)"]["candidates"].dropna(subset=["kwh_per_t"])
    fig.add_trace(go.Scatter(x=c["peak_temp_c"], y=c["kwh_per_t"], mode="markers", showlegend=False,
                             marker=dict(color=np.where(c["feasible"], green, grey), size=6), text=c["reason"]), 2, 2)
    br = opts["matt, summer (CW 28 degC)"]["best_row"]
    fig.add_trace(go.Scatter(x=[br["peak_temp_c"]], y=[br["kwh_per_t"]], mode="markers", name="optimum",
                             marker=dict(color=red, size=14, symbol="star")), 2, 2)
    fig.add_vline(x=44, line=dict(color=red, dash="dot"), row=2, col=2, annotation_text="limit - margin")
    fig.update_layout(title="Step 6 what-if simulator and optimiser (predictions from the simulated-data twin)", template="plotly_white",
                      height=820, font=dict(family="Arial"), legend=dict(orientation="h", y=-0.06))
    fig.update_xaxes(title_text="minutes", row=1, col=1); fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_xaxes(title_text="actual peak, degC", row=1, col=2); fig.update_yaxes(title_text="what-if peak, degC", row=1, col=2)
    fig.update_xaxes(title_text="HSD rpm", row=2, col=1); fig.update_yaxes(title_text="degC / kWh/t x3", row=2, col=1)
    fig.update_xaxes(title_text="predicted peak, degC", row=2, col=2); fig.update_yaxes(title_text="kWh/t", row=2, col=2)
    out = ROOT / "reports" / "step6_whatif.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"  written {out.relative_to(ROOT)}")

    banner("ALL STEP 6 CHECKS PASSED - ready for Step 7 (dashboard)")


if __name__ == "__main__":
    main()
