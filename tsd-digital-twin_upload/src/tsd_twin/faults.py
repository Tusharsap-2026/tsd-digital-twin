"""
faults.py - validation and tuning of the fault-detection rules (Step 8).

The twin never trains on fault labels. The simulator's hidden labels are used here ONLY to
score the hand-written rules and to pick their thresholds - exactly what a plant would do
with a few known events (a blade change, a jacket cleaning, a strainer blockage).

Contents
  indicator_table     one row per batch with the rule statistics exactly as the runtime rules
                      see them (rolling window + debounce), plus the hidden labels for scoring
  sweep_threshold     detection rate and false-alarm rate as a threshold moves (the ROC)
  choose_threshold    the operating point that meets the targets (detection >= X, false <= Y)
  confusion           what was flagged vs what was injected, incl. MULTIPLE-fault batches
  prealarm_scorecard  lead-time distribution, watch-flag analysis, breach-ETA accuracy
  drift_tracker       cross-batch CUSUM + trend on the per-batch indicators with a
                      remaining-life estimate ("wear alarm in about N batches")
  rule_cards          the plain-language card for every rule
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import runtime as rt
from .config import get

NAN = float("nan")
FAULTS = ["BLADE_WEAR", "JACKET_FOULING", "LOW_CW_FLOW", "OVERCHARGE"]


# ======================================================= rule statistics ==
def _rule_stat(x: np.ndarray, hold: int, direction: str) -> float:
    """The single number a debounced rule compares with its threshold.
    'below' rule fires iff some run of `hold` consecutive samples is all below thr
            <=> min over i of (max of the window ending at i) < thr   -> return that min
    'above' rule fires iff some run is all above thr                   -> return the max of window-mins"""
    s = pd.Series(x).dropna()
    if len(s) < hold:
        return NAN
    if direction == "below":
        return float(s.rolling(hold).max().min())
    return float(s.rolling(hold).min().max())


def indicator_table(ts: pd.DataFrame, batches: pd.DataFrame, cfg: dict, tm, thr: rt.Thresholds, base_lv: rt.Baselines,
                    verbose: bool = True) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the runtime over every batch and collect the rule statistics + labels."""
    rows, runs = [], {}
    for k, bid in enumerate(batches.index):
        run = rt.prepare_batch(ts[ts["batch_id"] == bid], cfg, tm, float(batches.loc[bid, "batch_mass_nominal_kg"]),
                               float(batches.loc[bid, "cw_flow_setpoint_m3h"]), thr, base_lv)
        runs[bid] = run
        hsd = run[(run["step"] == "high_speed_dispersion")]
        wear_rows = hsd[hsd["t_in_hsd_min"] >= thr.blade_wear_min_disp_min]
        foul_rows = hsd[(hsd["t_in_hsd_min"] >= thr.fouling_min_disp_min) & (hsd["flow_ratio"] >= thr.low_flow_ratio)]
        active = str(batches.loc[bid, "faults_active"]) if pd.notna(batches.loc[bid, "faults_active"]) else ""
        br = run["T_meas"] > run["limit_c"]
        pre = run["rule_TEMP_PREALARM"]
        t_br = run.loc[br, "t_min"]
        t_pre = run.loc[pre, "t_min"]
        eta_at_pre = run.loc[pre, "breach_eta_min"]
        rows.append({
            "batch_id": bid, "start_time": batches.loc[bid, "start_time"], "recipe_id": batches.loc[bid, "recipe_id"],
            "family": batches.loc[bid, "product_family"],
            # rule statistics (what the debounced rule compares)
            "stat_kw_resid": _rule_stat(wear_rows["kw_resid_pct"].values, thr.hold_rows, "below"),
            "stat_T_resid": _rule_stat(foul_rows["T_resid_replay"].values, thr.hold_rows, "above"),
            "stat_u_drop": _rule_stat(foul_rows["u_drop_pct"].values, thr.hold_rows, "above"),
            "stat_flow_ratio": _rule_stat(run["flow_ratio"].values, thr.hold_rows, "below"),
            "stat_mass_ratio": _rule_stat(run.loc[run["step"] != "charge_liquids", "mass_ratio"].values, thr.hold_rows, "above"),
            "grind_lag_last": float(run["grind_lag"].dropna().iloc[-1]) if run["grind_lag"].notna().any() else NAN,
            # per-batch medians for the drift tracker
            "kw_resid_median": float(hsd["kw_resid_pct"].median()), "u_drop_median": float(foul_rows["u_drop_pct"].median()) if len(foul_rows) else NAN,
            # what the current rules did
            "flag_BLADE_WEAR": bool(run["rule_BLADE_WEAR"].any()), "flag_JACKET_FOULING": bool(run["rule_JACKET_FOULING"].any()),
            "flag_LOW_CW_FLOW": bool(run["rule_LOW_CW_FLOW"].any()), "flag_OVERCHARGE": bool(run["rule_OVERCHARGE"].any()),
            "flag_TEMP_PREALARM": bool(pre.any()),
            # temperature outcome
            "breached": bool(br.any()), "peak_T": float(run["T_meas"].max()), "limit_c": float(run["limit_c"].iloc[0]),
            "t_breach_min": float(t_br.iloc[0]) if len(t_br) else NAN, "t_prealarm_min": float(t_pre.iloc[0]) if len(t_pre) else NAN,
            "eta_at_prealarm_min": float(eta_at_pre.iloc[0]) if len(eta_at_pre) else NAN,
            # hidden labels (scoring only)
            "true_wear": float(batches.loc[bid, "blade_wear"]), "true_fouling": float(batches.loc[bid, "jacket_fouling_m2k_w"]),
            "true_flow_frac": float(batches.loc[bid, "cw_flow_actual_frac"]), "true_mass_frac": float(batches.loc[bid, "overcharge_frac"]),
            "label_BLADE_WEAR": "BLADE_WEAR" in active, "label_JACKET_FOULING": "JACKET_FOULING" in active,
            "label_LOW_CW_FLOW": "LOW_CW_FLOW" in active, "label_OVERCHARGE": "OVERCHARGE" in active,
            "fault_injected": batches.loc[bid, "fault_injected"],
        })
        if verbose and (k + 1) % 100 == 0:
            print(f"  {k + 1} batches")
    return pd.DataFrame(rows), runs


# ======================================================= threshold sweep ==
def sweep_threshold(stat: pd.Series, label: pd.Series, direction: str, grid: np.ndarray,
                    clear: pd.Series | None = None) -> pd.DataFrame:
    """Detection rate on labelled batches and false-alarm rate on CLEARLY healthy batches
    (clear mask, default = not labelled) for every threshold in the grid."""
    clear = ~label if clear is None else clear
    rows = []
    for t in grid:
        fire = (stat < t) if direction == "below" else (stat > t)
        fire = fire.fillna(False)
        rows.append({"threshold": t, "detection_pct": 100 * fire[label].mean() if label.any() else NAN,
                     "false_pct": 100 * fire[clear].mean() if clear.any() else NAN,
                     "n_fired": int(fire.sum()), "n_labelled": int(label.sum()), "n_clear": int(clear.sum())})
    return pd.DataFrame(rows)


def choose_threshold(sw: pd.DataFrame, min_detection: float, max_false: float, direction: str, prefer: str = "conservative") -> dict:
    """Pick the threshold that meets both targets. prefer='conservative': among the best-scoring
    feasible points, the one that fires least (right for slowly drifting faults, where a looser
    threshold only adds early flags). prefer='midpoint': the middle of the feasible band (right for
    step faults such as a blocked strainer, where the simulated cases only sample one severity and
    the edge of the band would be an artefact of that). If nothing is feasible: max false alarm is
    respected first, since a noisy alarm gets switched off."""
    ok = sw[(sw["detection_pct"] >= min_detection) & (sw["false_pct"] <= max_false)]
    if not ok.empty:
        if prefer == "midpoint":
            mid = ok["threshold"].median()
            best = ok.iloc[(ok["threshold"] - mid).abs().argsort().iloc[0]]
        else:
            top = ok[ok["detection_pct"] >= ok["detection_pct"].max() - 1e-9]
            top = top[top["false_pct"] <= top["false_pct"].min() + 1e-9]
            best = top.iloc[0] if direction == "below" else top.iloc[-1]       # fires least
        return {"threshold": float(best["threshold"]), "detection_pct": float(best["detection_pct"]), "false_pct": float(best["false_pct"]),
                "meets_targets": True, "prefer": prefer}
    within_false = sw[sw["false_pct"] <= max_false]
    pick = within_false.sort_values("detection_pct", ascending=False).iloc[0] if not within_false.empty else sw.sort_values("false_pct").iloc[0]
    return {"threshold": float(pick["threshold"]), "detection_pct": float(pick["detection_pct"]), "false_pct": float(pick["false_pct"]),
            "meets_targets": False}


def tune_all(ind: pd.DataFrame, cfg: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Sweep every rule's threshold and choose operating points against the KPI targets."""
    min_det = get(cfg, "kpi_targets.min_fault_detection_rate_pct")
    max_false = get(cfg, "kpi_targets.max_false_alarms_per_100_batches")
    sweeps, chosen = {}, {}
    # clearly healthy masks: wear < 0.10 (not the grey zone 0.10-0.18), fouling < 0.0003, flow = 1, mass <= 1.02
    clear_wear = ind["true_wear"] < 0.10
    clear_foul = ind["true_fouling"] < 0.0003
    specs = [
        ("BLADE_WEAR", "stat_kw_resid", "below", np.arange(-8.0, -0.5, 0.25), ind["label_BLADE_WEAR"], clear_wear, "conservative"),
        ("JACKET_FOULING_u", "stat_u_drop", "above", np.arange(2.0, 30.0, 0.5), ind["label_JACKET_FOULING"], clear_foul, "conservative"),
        ("JACKET_FOULING_T", "stat_T_resid", "above", np.arange(0.2, 3.0, 0.05), ind["label_JACKET_FOULING"], clear_foul, "conservative"),
        ("LOW_CW_FLOW", "stat_flow_ratio", "below", np.arange(0.5, 1.0, 0.01), ind["label_LOW_CW_FLOW"], ~ind["label_LOW_CW_FLOW"], "midpoint"),
        ("OVERCHARGE", "stat_mass_ratio", "above", np.arange(1.005, 1.08, 0.0025), ind["label_OVERCHARGE"], ~ind["label_OVERCHARGE"], "midpoint"),
    ]
    for name, col, direction, grid, label, clear, prefer in specs:
        sw = sweep_threshold(ind[col], label, direction, grid, clear)
        sweeps[name] = sw
        chosen[name] = choose_threshold(sw, min_det, max_false, direction, prefer)
        chosen[name].update({"stat": col, "direction": direction, "n_labelled": int(label.sum()), "n_clear": int(clear.sum())})
    # fouling: pick the rule VARIANT (U only, T only, U or T) that meets the targets with the highest detection
    lab, clear = ind["label_JACKET_FOULING"], clear_foul
    fire_u = (ind["stat_u_drop"] > chosen["JACKET_FOULING_u"]["threshold"]).fillna(False)
    fire_T = (ind["stat_T_resid"] > chosen["JACKET_FOULING_T"]["threshold"]).fillna(False)
    variants = {"u": fire_u, "T": fire_T, "u_or_T": fire_u | fire_T, "u_and_T": fire_u & fire_T}
    table = pd.DataFrame([{"mode": m, "detection_pct": 100 * f[lab].mean(), "false_pct": 100 * f[clear].mean()} for m, f in variants.items()])
    ok = table[(table["detection_pct"] >= min_det) & (table["false_pct"] <= max_false)]
    pick = (ok if not ok.empty else table[table["false_pct"] <= max_false] if (table["false_pct"] <= max_false).any() else table)
    best = pick.sort_values(["detection_pct", "false_pct"], ascending=[False, True]).iloc[0]
    chosen["JACKET_FOULING"] = {"mode": best["mode"], "detection_pct": float(best["detection_pct"]), "false_pct": float(best["false_pct"]),
                                "meets_targets": bool(best["detection_pct"] >= min_det and best["false_pct"] <= max_false), "variants": table}
    return chosen, sweeps


def apply_thresholds(ind: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """Re-evaluate the batch-level flags with the chosen thresholds (fouling = U-rule OR T-rule)."""
    out = ind.copy()
    out["tuned_BLADE_WEAR"] = (ind["stat_kw_resid"] < chosen["BLADE_WEAR"]["threshold"]).fillna(False)
    fire_u = (ind["stat_u_drop"] > chosen["JACKET_FOULING_u"]["threshold"]).fillna(False)
    fire_T = (ind["stat_T_resid"] > chosen["JACKET_FOULING_T"]["threshold"]).fillna(False)
    mode = chosen.get("JACKET_FOULING", {}).get("mode", "u_or_T")
    out["tuned_JACKET_FOULING"] = {"u": fire_u, "T": fire_T, "u_or_T": fire_u | fire_T, "u_and_T": fire_u & fire_T}[mode]
    out["tuned_LOW_CW_FLOW"] = (ind["stat_flow_ratio"] < chosen["LOW_CW_FLOW"]["threshold"]).fillna(False)
    out["tuned_OVERCHARGE"] = (ind["stat_mass_ratio"] > chosen["OVERCHARGE"]["threshold"]).fillna(False)
    return out


# ============================================================ confusion ==
def confusion(ind: pd.DataFrame, prefix: str = "tuned_") -> pd.DataFrame:
    """Rows: injected fault class (incl. MULTIPLE and NONE). Columns: share flagged with each rule."""
    g = ind.groupby("fault_injected")
    tab = pd.DataFrame({"n": g.size()})
    for f in FAULTS:
        tab[f"flag_{f}_pct"] = (100 * g[f"{prefix}{f}"].mean()).round(0)
    tab["no_flag_pct"] = (100 * g.apply(lambda d: (~d[[f"{prefix}{f}" for f in FAULTS]].any(axis=1)).mean())).round(0)
    return tab


def per_fault_scorecard(ind: pd.DataFrame, prefix: str = "tuned_") -> pd.DataFrame:
    rows = []
    clear = {"BLADE_WEAR": ind["true_wear"] < 0.10, "JACKET_FOULING": ind["true_fouling"] < 0.0003,
             "LOW_CW_FLOW": ~ind["label_LOW_CW_FLOW"], "OVERCHARGE": ~ind["label_OVERCHARGE"]}
    grey = {"BLADE_WEAR": (ind["true_wear"] >= 0.10) & ~ind["label_BLADE_WEAR"],
            "JACKET_FOULING": (ind["true_fouling"] >= 0.0003) & ~ind["label_JACKET_FOULING"]}
    for f in FAULTS:
        lab, flg = ind[f"label_{f}"], ind[f"{prefix}{f}"]
        rows.append({"fault": f, "labelled": int(lab.sum()), "caught": int((lab & flg).sum()), "detection_pct": 100 * (lab & flg).sum() / max(lab.sum(), 1),
                     "clearly_healthy": int(clear[f].sum()), "false_flags": int((clear[f] & flg).sum()),
                     "false_per_100": 100 * (clear[f] & flg).sum() / max(clear[f].sum(), 1),
                     "grey_zone": int(grey[f].sum()) if f in grey else 0,
                     "grey_flagged": int((grey[f] & flg).sum()) if f in grey else 0})
    return pd.DataFrame(rows)


# ============================================================ pre-alarm ==
def prealarm_scorecard(ind: pd.DataFrame, cfg: dict) -> dict:
    lead_target = get(cfg, "kpi_targets.breach_lead_time_min")
    br = ind[ind["breached"]]
    lead = br["t_breach_min"] - br["t_prealarm_min"]
    watch = ind[ind["flag_TEMP_PREALARM"] & ~ind["breached"]]
    margin_watch = watch["limit_c"] - watch["peak_T"]
    quiet = ind[~ind["flag_TEMP_PREALARM"] & ~ind["breached"]]
    eta_err = (br["eta_at_prealarm_min"] - lead).dropna()
    return {"breaches": int(len(br)), "prealarmed": int(lead.notna().sum()), "lead_median_min": float(lead.median()), "lead_min_min": float(lead.min()),
            "lead_p10_min": float(lead.quantile(0.1)), "lead_target_met_pct": float(100 * (lead >= lead_target).mean()),
            "watch_flags": int(len(watch)), "watch_peak_within_1c": int((margin_watch <= 1.0).sum()),
            "watch_peak_within_2c": int((margin_watch <= 2.0).sum()), "watch_margin_median_c": float(margin_watch.median()) if len(watch) else NAN,
            "quiet_batches": int(len(quiet)), "quiet_closest_margin_c": float((quiet["limit_c"] - quiet["peak_T"]).min()) if len(quiet) else NAN,
            "eta_at_prealarm_available": int(br["eta_at_prealarm_min"].notna().sum()),
            "eta_error_median_min": float(eta_err.median()) if len(eta_err) else NAN,
            "eta_abs_error_median_min": float(eta_err.abs().median()) if len(eta_err) else NAN}


# ========================================================= drift tracker ==
def drift_tracker(series: pd.Series, baseline: float, direction: str, slack: float, h: float, window: int,
                  alarm_level: float) -> pd.DataFrame:
    """Cross-batch CUSUM and rolling linear trend on a per-batch indicator.
    direction 'down': drift is a fall below baseline (power at speed); 'up': a rise (U drop).
    alarm_level: the per-batch rule threshold, used for the remaining-batches estimate."""
    x = series.values.astype(float)
    n = len(x)
    S = np.zeros(n)
    slope = np.full(n, NAN)
    remaining = np.full(n, NAN)
    cusum_alarm = np.zeros(n, dtype=bool)
    healthy_run = 0
    for i in range(n):
        xi = x[i] if not np.isnan(x[i]) else baseline
        dev = (baseline - xi) if direction == "down" else (xi - baseline)
        healthy_run = healthy_run + 1 if dev < 0 else 0
        prev = S[i - 1] if i else 0.0
        if healthy_run >= 3:                 # back on the healthy side for 3 batches: a maintenance event, restart
            prev = 0.0
        S[i] = max(0.0, prev + dev - slack)
        cusum_alarm[i] = S[i] > h
        lo = max(0, i - window + 1)
        seg = x[lo:i + 1]
        ok = ~np.isnan(seg)
        if ok.sum() >= max(8, window // 3):
            t = np.arange(lo, i + 1)[ok]
            b1, b0 = np.polyfit(t, seg[ok], 1)
            slope[i] = b1
            level = b0 + b1 * i
            if direction == "down" and b1 < -1e-6 and level > alarm_level:
                remaining[i] = (level - alarm_level) / (-b1)
            elif direction == "up" and b1 > 1e-6 and level < alarm_level:
                remaining[i] = (alarm_level - level) / b1
    return pd.DataFrame({"value": x, "cusum": S, "cusum_alarm": cusum_alarm, "slope_per_batch": slope,
                         "batches_to_alarm": remaining}, index=series.index)


# ======================================================= config write-back ==
def write_thresholds_to_config(path, updates: dict) -> list[str]:
    """Update 'value' (and 'note') of keys in the alerts section of assumptions.yaml in place,
    keeping every comment and the rest of the file untouched. updates: {key: (value, note)}."""
    from pathlib import Path
    lines = Path(path).read_text().splitlines(keepends=True)
    changed = []
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("alerts:"))
    except StopIteration:
        raise ValueError("no alerts section in the config")
    i = start + 1
    blocks = {}
    while i < len(lines) and (lines[i].startswith("  ") or lines[i].strip() == "" or lines[i].startswith("#")):
        if lines[i].startswith("  ") and not lines[i].startswith("    ") and lines[i].strip().endswith(":"):
            key = lines[i].strip()[:-1]
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                j += 1
            blocks[key] = (i, j)
            i = j
        else:
            i += 1
    for key, (value, note) in updates.items():
        if key not in blocks:
            continue
        a, b = blocks[key]
        has_note = False
        for k in range(a + 1, b):
            if lines[k].startswith("    value:"):
                v = f'"{value}"' if isinstance(value, str) else (str(value) if not isinstance(value, float) else f"{value:g}")
                lines[k] = f"    value: {v}\n"
            if lines[k].startswith("    note:") and note is not None:
                lines[k] = f'    note: "{note}"\n'
                has_note = True
        if note is not None and not has_note:
            lines.insert(b, f'    note: "{note}"\n')
            for kk in blocks:                       # shift later blocks
                a2, b2 = blocks[kk]
                if a2 > a:
                    blocks[kk] = (a2 + 1, b2 + 1)
            blocks[key] = (a, b + 1)
        changed.append(key)
    Path(path).write_text("".join(lines))
    return changed


# ============================================================ rule cards ==
def rule_cards(chosen: dict, score: pd.DataFrame, pre: dict, thr: rt.Thresholds, base_lv: rt.Baselines, cfg: dict) -> str:
    sc = score.set_index("fault")
    lead_target = get(cfg, "kpi_targets.breach_lead_time_min")

    def line(f):
        r = sc.loc[f]
        return (f"Achieved on the simulated year: {r['caught']:.0f} of {r['labelled']:.0f} labelled batches caught ({r['detection_pct']:.0f} %), "
                f"{r['false_flags']:.0f} false flags on {r['clearly_healthy']:.0f} clearly healthy batches ({r['false_per_100']:.1f} per 100).")

    cards = f"""# TSD twin - alert rule cards (Step 8)

Every rule works on MEASURED tags and the gap between the measured and the physics-predicted
value. "Normal level" means the gap seen on healthy batches, learned from {base_lv.n_batches} batches
(models/runtime_calibration.json): HSD power {base_lv.kw_resid_pct:+.1f} % vs physics, product
{base_lv.T_resid_replay_c:+.1f} degC vs measured-power physics, implied U {100 * base_lv.u_ratio:.0f} % of the
physics U. Every number below comes from SIMULATED data; refit the normal levels and re-tune the
thresholds on the first real batches (Step 9).

---

## TEMP_PREALARM - product temperature will reach the limit
**Watches:** the 5-min-ahead temperature forecast (physics + residual model + the batch's last 5 min)
and the measured temperature.
**Fires when:** forecast or measured temperature > limit - {get(cfg, 'products.water_based.temp_alarm_margin_c')} degC, held {max(thr.hold_rows // 2, 1) * 10} s.
**Also reports:** breach ETA = first crossing of the forecast curve shifted by the current residual
(fading with a {thr.decay_min:.0f}-min time constant), and the likely cause (see evidence list).
**Achieved:** {pre['prealarmed']} of {pre['breaches']} breaches pre-alarmed, lead time median {pre['lead_median_min']:.0f} min,
minimum {pre['lead_min_min']:.0f} min (target >= {lead_target}); {pre['watch_flags']} watch flags on batches that stayed under the limit,
of which {pre['watch_peak_within_2c']} peaked within 2 degC of it.
**Operator response:** reduce HSD speed 50-100 rpm or open the chilled-water valve; check the inlet temperature.
**Would fire wrongly on real data if:** the RTD is out of calibration (cross-check the now-cast gap), or the
recipe's limit in MES is wrong.

## LOW_CW_FLOW - chilled-water flow below set-point
**Watches:** measured flow / set-point.  **Fires when:** < {100 * chosen['LOW_CW_FLOW']['threshold']:.0f} % for {thr.hold_rows * 10} s.
{line('LOW_CW_FLOW')}
**Operator response:** check the strainer and the control valve; expect a hotter batch - the twin re-forecasts.
**Would fire wrongly if:** the flow meter drifts or the set-point tag is not updated when the operator changes it.

## OVERCHARGE - more material in the vessel than the recipe says
**Watches:** load-cell mass / recipe nominal (after charging).  **Fires when:** > {100 * chosen['OVERCHARGE']['threshold']:.1f} % for {thr.hold_rows * 10} s.
{line('OVERCHARGE')}
**Operator response:** confirm the dosing record; a 7-15 % overcharge lowers kWh/t but lengthens the grind and
can exceed the working volume.
**Would fire wrongly if:** the load cells are not tared between batches or the nominal mass in MES is stale.

## BLADE_WEAR - sawtooth blade losing its edge
**Watches:** HSD electrical power vs the physics power at the ACTUAL speed and viscosity, rolled over {thr.kw_window_min:.0f} min,
relative to the normal level.  **Fires when:** more than {abs(chosen['BLADE_WEAR']['threshold']):.1f} % below normal for {thr.hold_rows * 10} s,
after {thr.blade_wear_min_disp_min:.0f} min of high-speed dispersion.  **Corroboration:** lab grind behind the physics prediction.
{line('BLADE_WEAR')} Batches in the grey zone (wear 10-18 %, not yet labelled) flagged: {sc.loc['BLADE_WEAR', 'grey_flagged']:.0f} of
{sc.loc['BLADE_WEAR', 'grey_zone']:.0f} - early warnings, not errors.
**Cross-batch tracker:** CUSUM on the per-batch power residual with slack {thr.drift_slack} % and decision interval {thr.drift_h} %,
plus a {thr.drift_window}-batch trend giving "batches until the alarm level".
**Operator response:** plan a blade inspection; expect longer grinds and more extensions until it is changed.
**Would fire wrongly if:** the viscosity lot is unusually thin (check the anchor power residual), the VFD power
estimate drifts, or the blade is run at a different height.

## JACKET_FOULING - jacket transferring less heat than normal
**Watches:** (a) the implied overall U from the measured jacket duty, Q = m_dot cp (T_out - T_in), inverted through the
jacket equation, vs the physics U, relative to the normal level - only when the product is >= {thr.fouling_min_dt_c:.0f} degC above the
inlet; (b) the product temperature vs the measured-power physics, relative to the normal gap.
**Fires when:** rule variant '{chosen['JACKET_FOULING']['mode']}' - (a) > {chosen['JACKET_FOULING_u']['threshold']:.1f} % U drop, (b) > {chosen['JACKET_FOULING_T']['threshold']:.2f} degC -
held {thr.hold_rows * 10} s, after {thr.fouling_min_disp_min:.0f} min of dispersion, with flow normal (so it is not LOW_CW_FLOW).
Variant chosen because it meets the targets with the highest detection; the other variant stays as evidence in the cause text.
{line('JACKET_FOULING')}
**Cross-batch tracker:** CUSUM on the per-batch U drop; trend gives "batches until the alarm level".
**Operator response:** schedule a jacket chemical clean; until then plan summer batches with the what-if tool.
**Would fire wrongly if:** the jacket outlet RTD reads low, the flow meter reads high, or the inlet cycles strongly
within the averaging window.
"""
    return cards
