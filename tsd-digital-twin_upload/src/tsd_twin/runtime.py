"""
runtime.py - the per-batch engine the dashboard replays and the live twin will run (Step 7).

For one batch of measured rows (the canonical schema) it produces, row by row:
  * physics baselines (open and measured-power replay) and the Step 5 predictions:
      T_nowcast  what the twin thinks the temperature is right now (sensor cross-check)
      T_ahead    what it predicted 5 min ago for now  (scored live: ahead-error)
      T_next5    what it predicts now for 5 min from now
  * a limit-breach ETA: the hybrid forecast curve for the rest of the batch, shifted by the
    current residual (measured - forecast) which is assumed to fade with a time constant
    (config alerts.residual_persistence_decay_min). Explainable and good enough for a
    several-minutes warning; Step 8 scores it.
  * fault indicators from measured tags and physics residuals only:
      kw_resid_pct     HSD power vs physics at the actual speed  (blade wear: negative)
      T_resid_replay   measured T vs physics driven by measured power (jacket side: positive)
      flow_ratio       measured flow / set-point                     (low flow)
      mass_ratio       load cells / recipe nominal                   (overcharge)
      cw_inlet_c       utility cause
  * alert codes with a debounce, and a 'likely cause' text with the evidence.

Everything is causal: row i uses rows <= i only (the one exception is a 10 s central
difference in the physics-side slope, which is harmless and noted).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as ft, models as md, physics as ph
from .config import get

NAN = float("nan")


@dataclass
class Thresholds:
    horizon_min: float
    projection_max_min: float
    decay_min: float
    kw_window_min: float
    blade_wear_kw_pct: float
    blade_wear_min_disp_min: float
    fouling_T_resid_c: float
    fouling_min_disp_min: float
    fouling_u_drop_pct: float
    fouling_mode: str
    fouling_min_dt_c: float
    u_window_min: float
    drift_window: int
    drift_slack: float
    drift_h: float
    low_flow_ratio: float
    overcharge_ratio: float
    high_cw_inlet_c: float
    hold_rows: int

    @classmethod
    def from_config(cls, cfg: dict) -> "Thresholds":
        a = "alerts"
        return cls(get(cfg, f"{a}.horizon_min"), get(cfg, f"{a}.projection_max_min"), get(cfg, f"{a}.residual_persistence_decay_min"),
                   get(cfg, f"{a}.kw_resid_window_min"), get(cfg, f"{a}.blade_wear_kw_resid_pct"), get(cfg, f"{a}.blade_wear_min_dispersion_min"),
                   get(cfg, f"{a}.fouling_T_resid_replay_c"), get(cfg, f"{a}.fouling_min_dispersion_min"),
                   get(cfg, f"{a}.fouling_u_drop_pct"), str(get(cfg, f"{a}.fouling_rule_mode")), get(cfg, f"{a}.fouling_min_product_jacket_dt_c"),
                   get(cfg, f"{a}.u_window_min"),
                   int(get(cfg, f"{a}.drift_window_batches")), get(cfg, f"{a}.drift_cusum_slack"), get(cfg, f"{a}.drift_cusum_h"),
                   get(cfg, f"{a}.low_flow_ratio"),
                   get(cfg, f"{a}.overcharge_ratio"), get(cfg, f"{a}.high_cw_inlet_c"), int(get(cfg, f"{a}.alert_hold_rows")))


@dataclass
class Baselines:
    """The NORMAL gap between plant and physics, learned from healthy-majority training batches.
    Fault rules work on the residual minus this baseline: a fault is a change from normal, not a
    difference from the textbook physics. Both numbers are refitted first on real batches (Step 9)."""
    kw_resid_pct: float = 0.0          # median HSD power residual in high-speed dispersion
    T_resid_replay_c: float = 0.0      # median temperature residual vs measured-power physics
    u_ratio: float = 1.0               # median implied-U / physics-U in high-speed dispersion (clean jacket, normal flow)
    n_batches: int = 0

    @classmethod
    def load(cls, path: Path | str) -> "Baselines":
        path = Path(path)
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(float(d.get("kw_resid_pct", 0.0)), float(d.get("T_resid_replay_c", 0.0)), float(d.get("u_ratio", 1.0)),
                   int(d.get("n_batches", 0)))

    def save(self, path: Path | str, note: str = "") -> None:
        Path(path).write_text(json.dumps({"kw_resid_pct": self.kw_resid_pct, "T_resid_replay_c": self.T_resid_replay_c,
                                          "u_ratio": self.u_ratio, "n_batches": self.n_batches, "note": note}, indent=2))


def _ahead_frame(b: pd.DataFrame, base: pd.DataFrame, f: pd.DataFrame, cfg: dict, recipe, lag: int) -> pd.DataFrame:
    """FEATURES_T_AHEAD for every row i, describing the state `lag` samples AHEAD of i: the open
    physics continued past the last received row, with the lag_* columns taken from row i."""
    dt = float(get(cfg, "data.sample_period_s"))
    p = ph.PhysicsParams(cfg, recipe.family)
    comp = ph.composition(recipe, p)
    phases = ft.phases_from_measured(b, dt, comp)
    last = phases[-1]
    ext = lag * dt
    scale = (last.duration_s + ext) / max(last.duration_s, dt)
    phases[-1] = ph.Phase(last.name, last.duration_s + ext, last.hsd_rpm, last.anchor_rpm, last.mass_added_kg * scale, last.added_kind)
    cond = ft.conditions_from_measured(b, replay_power=False)
    df_ext, _ = ph.simulate_batch(recipe, phases, cond, cfg)
    n = len(b)
    idx = np.arange(n) + lag
    idx = np.minimum(idx, len(df_ext) - 1)
    tgt = df_ext.iloc[idx].reset_index(drop=True)
    g = pd.DataFrame(index=b.index)
    g["T_phys_open"] = tgt["T_product_c"].values
    g["dT_phys_open_to_jacket_c"] = tgt["T_product_c"].values - tgt["T_jacket_in_c"].values
    g["dTdt_phys_open_c_per_min"] = ft.backward_slope(df_ext["T_product_c"].values, dt)[idx]
    g["mu_phys_open"] = tgt["mu_pa_s"].values
    g["U_phys_open"] = tgt["U_w_m2k"].values
    g["A_phys_m2"] = tgt["A_wetted_m2"].values
    g["Q_jacket_phys_open_kw"] = tgt["Q_jacket_kw"].values
    g["hegman_phys"] = tgt["hegman_model"].values
    g["submergence_phys"] = tgt["submergence"].values
    g["hsd_kw_phys"] = tgt["hsd_kw"].values
    g["anchor_kw_phys"] = tgt["anchor_kw"].values
    g["energy_phys_cum_kwh"] = tgt["energy_cum_kwh"].values
    g["t_min"] = tgt["t_s"].values / 60.0
    g["step_idx"] = tgt["step"].map(ft.STEP_IDX).values
    g["t_in_step_min"] = df_ext.groupby("step").cumcount().values[idx] * dt / 60.0
    for c in ["family_wb", "solids_pct", "pvc_pct", "phi_millbase", "cp_final_kj_kgk", "density_final_kg_l", "batch_mass_final_kg"]:
        g[c] = f[c].values
    g["mass_frac_of_final"] = tgt["batch_mass_kg"].values / recipe.batch_mass_kg
    g["hsd_rpm_setpoint"] = tgt["hsd_rpm_setpoint"].values
    g["anchor_rpm"] = tgt["anchor_rpm"].values
    g["T_jacket_in_c"] = tgt["T_jacket_in_c"].values
    g["cw_flow_m3h"] = tgt["cw_flow_m3h"].values
    g["T_ambient_c"] = tgt["T_ambient_c"].values
    # the batch's own last 5 minutes = now
    T_meas = b["T_product_c"].values
    g["lag_T_meas_c"] = T_meas
    g["lag_resid_replay_c"] = T_meas - base["T_phys_replay"].values
    g["lag_resid_open_c"] = T_meas - base["T_phys_open"].values
    g["lag_dTdt_meas_c_per_min"] = pd.Series(ft.backward_slope(T_meas, dt)).rolling(int(120 / dt), min_periods=1).mean().values
    g["lag_T_phys_replay_c"] = base["T_phys_replay"].values
    return g


def _fouling_condition(out: pd.DataFrame, thr: "Thresholds") -> np.ndarray:
    u = (out["u_drop_pct"] > thr.fouling_u_drop_pct).values
    T = (out["T_resid_replay"] > thr.fouling_T_resid_c).values
    return {"u": u, "T": T, "u_or_T": u | T, "u_and_T": u & T}.get(thr.fouling_mode, u | T)


def _hold(mask: np.ndarray, n: int) -> np.ndarray:
    """True where the condition has held for n consecutive samples (debounce)."""
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run >= n
    return out


def prepare_batch(b: pd.DataFrame, cfg: dict, tm: md.TemperatureModels | None, nominal_mass_kg: float,
                  cw_flow_setpoint_m3h: float, thr: Thresholds | None = None, base_lv: Baselines | None = None) -> pd.DataFrame:
    """Run the twin over one batch of measured rows. Returns one row per sample with the
    measured tags, baselines, predictions, breach ETA, indicators, alert code and cause."""
    thr = thr or Thresholds.from_config(cfg)
    base_lv = base_lv or Baselines()
    b = b.reset_index(drop=True)
    dt = float(get(cfg, "data.sample_period_s"))
    lag = int(round(thr.horizon_min * 60.0 / dt))
    recipe = ft.recipe_from_measured(b, nominal_mass_kg)
    p = ph.PhysicsParams(cfg, recipe.family)
    limit = p.T_limit
    margin = get(cfg, f"products.{recipe.family}.temp_alarm_margin_c")

    base, _ = ft.physics_baselines(b, cfg, batch_mass_kg=nominal_mass_kg)
    f = ft.row_features_for_batch(b, base, cfg, thr.horizon_min, batch_mass_kg=nominal_mass_kg)
    out = pd.DataFrame({"batch_id": b["batch_id"], "timestamp": b["timestamp"], "t_s": b["t_s"], "t_min": b["t_s"] / 60.0,
                        "step": b["step"], "T_meas": b["T_product_c"], "T_phys_open": base["T_phys_open"],
                        "T_phys_replay": base["T_phys_replay"], "hsd_kw": b["hsd_kw"], "hsd_kw_phys": base["hsd_kw_phys"],
                        "anchor_kw": b["anchor_kw"], "hsd_rpm": b["hsd_rpm"], "hsd_rpm_setpoint": b["hsd_rpm_setpoint"],
                        "T_jacket_in": b["T_jacket_in_c"], "T_jacket_out": b["T_jacket_out_c"], "cw_flow": b["cw_flow_m3h"],
                        "batch_mass_kg": b["batch_mass_kg"], "hegman_lab": b["hegman"], "hegman_phys": base["hegman_phys"],
                        "energy_cum_kwh": f["energy_cum_kwh"], "specific_energy_cum_kwh_t": f["specific_energy_cum_kwh_t"],
                        "cooling_duty_meas_kw": f["cooling_duty_meas_kw"], "Q_jacket_phys_kw": base["Q_jacket_phys_kw"],
                        "limit_c": limit, "prealarm_c": limit - margin})

    # ---- predictions
    if tm is not None:
        pred = tm.predict(f)
        out["T_forecast"] = pred["T_hybrid_forecast"].values      # recipe/set-points only: the pre-batch curve
        out["T_nowcast"] = pred["T_hybrid_nowcast"].values
        out["T_ahead"] = pred["T_ahead"].values                   # made 5 min ago for this row
        # prediction made NOW for 5 min from now: physics states of the TARGET time from an open
        # physics run extended `lag` samples beyond the rows received (the current phase continues
        # at its set-points), recipe/set-points as now, lag features = now
        g_next = _ahead_frame(b, base, f, cfg, recipe, lag)
        valid = g_next[md.FEATURES_T_AHEAD].notna().all(axis=1).values
        n = len(b)
        nxt = np.full(n, NAN)
        if valid.any():
            nxt[valid] = tm.predict_ahead(g_next.loc[valid])
        out["T_next5"] = nxt
    else:
        out["T_forecast"] = out["T_nowcast"] = out["T_ahead"] = out["T_phys_open"]
        out["T_next5"] = out["T_phys_open"].shift(-lag)

    # ---- breach ETA: forecast curve for the rest of the batch + fading current residual
    T_fc = out["T_forecast"].values
    T_meas = out["T_meas"].values
    n = len(b)
    k_max = int(thr.projection_max_min * 60.0 / dt)
    tau = np.arange(1, k_max + 1) * dt / 60.0
    fade = np.exp(-tau / thr.decay_min)
    eta = np.full(n, NAN)
    peak_proj = np.full(n, NAN)
    for i in range(n):
        j_end = min(n, i + 1 + k_max)
        if j_end <= i + 1:
            continue
        resid = T_meas[i] - T_fc[i]
        proj = T_fc[i + 1:j_end] + resid * fade[: j_end - i - 1]
        peak_proj[i] = max(T_meas[i], proj.max())
        over = np.where(proj > limit)[0]
        if over.size:
            eta[i] = (over[0] + 1) * dt / 60.0
    out["T_peak_projected"] = peak_proj
    out["breach_eta_min"] = eta

    # ---- fault indicators (measured tags and residuals only)
    in_hsd = (b["step"] == "high_speed_dispersion").values
    t_in_hsd = np.where(in_hsd, b.groupby("step").cumcount().values * dt / 60.0, NAN)
    kw_resid_pct = 100.0 * (b["hsd_kw"] - base["hsd_kw_phys"]) / base["hsd_kw_phys"].clip(lower=1.0)
    win = max(int(thr.kw_window_min * 60.0 / dt), 1)
    out["kw_resid_pct_raw"] = kw_resid_pct.where(in_hsd).rolling(win, min_periods=max(win // 2, 1)).mean().values
    out["T_resid_replay_raw"] = (b["T_product_c"] - base["T_phys_replay"]).rolling(win, min_periods=1).mean().values
    # relative to the NORMAL gap: this is what the fault rules look at
    out["kw_resid_pct"] = out["kw_resid_pct_raw"] - base_lv.kw_resid_pct
    out["T_resid_replay"] = out["T_resid_replay_raw"] - base_lv.T_resid_replay_c
    out["T_resid_nowcast"] = (out["T_meas"] - out["T_nowcast"]).rolling(win, min_periods=1).mean().values
    # implied overall heat-transfer coefficient from the MEASURED jacket duty:
    #   Q = UA dT / (1 + UA / 2C)  ->  UA = Q / (dT - Q / 2C),  C = m_dot cp of the water [W/K]
    rho_w, cp_w = get(cfg, "equipment.cooling_jacket.water_density_kg_m3"), get(cfg, "equipment.cooling_jacket.water_cp_kj_kgk")
    C = b["cw_flow_m3h"].values * rho_w / 3600.0 * cp_w * 1000.0
    Q = out["cooling_duty_meas_kw"].values * 1000.0
    dT = b["T_product_c"].values - b["T_jacket_in_c"].values
    denom = dT - Q / (2.0 * np.maximum(C, 1.0))
    ua_impl = np.where((dT >= thr.fouling_min_dt_c) & (denom > 0.5), Q / np.maximum(denom, 0.5), NAN)
    ua_phys = (base["U_phys"] * base["A_phys_m2"]).values
    u_ratio = pd.Series(ua_impl / np.maximum(ua_phys, 1.0))
    uwin = max(int(thr.u_window_min * 60.0 / dt), 1)
    out["u_ratio_raw"] = u_ratio.rolling(uwin, min_periods=max(uwin // 2, 1)).median().values
    out["u_drop_pct"] = 100.0 * (1.0 - out["u_ratio_raw"] / max(base_lv.u_ratio, 1e-6))      # positive = jacket transferring less than normal
    out["flow_ratio"] = b["cw_flow_m3h"].values / max(cw_flow_setpoint_m3h, 0.1)
    out["mass_ratio"] = b["batch_mass_kg"].values / max(nominal_mass_kg, 1.0)
    out["t_in_hsd_min"] = t_in_hsd
    lab = b["hegman"].ffill()
    out["grind_lag"] = (base["hegman_phys"] - lab).where(b["hegman"].notna()).ffill().values   # physics ahead of lab -> slow grind

    # ---- rules with debounce
    rules = {
        "LOW_CW_FLOW": _hold((out["flow_ratio"] < thr.low_flow_ratio).values, thr.hold_rows),
        "OVERCHARGE": _hold((out["mass_ratio"] > thr.overcharge_ratio).values & (b["step"] != "charge_liquids").values, thr.hold_rows),
        "BLADE_WEAR": _hold((out["kw_resid_pct"] < thr.blade_wear_kw_pct).values & in_hsd & (t_in_hsd >= thr.blade_wear_min_disp_min), thr.hold_rows),
        "JACKET_FOULING": _hold(_fouling_condition(out, thr) & in_hsd & (t_in_hsd >= thr.fouling_min_disp_min)
                                & (out["flow_ratio"] >= thr.low_flow_ratio).values, thr.hold_rows),
        "TEMP_PREALARM": _hold((out["T_next5"] > limit - margin).values | (out["T_meas"] > limit - margin).values, max(thr.hold_rows // 2, 1)),
        "TEMP_LIMIT": (out["T_meas"] > limit).values,
    }
    for k, v in rules.items():
        out[f"rule_{k}"] = v
    priority = ["TEMP_LIMIT", "TEMP_PREALARM", "LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"]
    code = np.full(n, "NONE", dtype=object)
    for k in reversed(priority):
        code[rules[k]] = k
    out["alert_code"] = code

    # ---- likely cause of a high temperature, from the evidence available at that moment
    causes = []
    for i in range(n):
        if code[i] not in ("TEMP_LIMIT", "TEMP_PREALARM"):
            causes.append("")
            continue
        ev = []
        if rules["LOW_CW_FLOW"][i]:
            ev.append(f"cooling-water flow {out['cw_flow'].iloc[i]:.1f} m3/h = {100 * out['flow_ratio'].iloc[i]:.0f} % of set-point")
        if rules["JACKET_FOULING"][i]:
            ev.append(f"jacket transferring {out['u_drop_pct'].iloc[i]:.0f} % less heat than normal; product {out['T_resid_replay'].iloc[i]:+.1f} degC "
                      f"above its normal gap to the measured-power physics")
        if out["T_jacket_in"].iloc[i] > thr.high_cw_inlet_c:
            ev.append(f"chilled-water inlet {out['T_jacket_in'].iloc[i]:.1f} degC (utility)")
        if rules["OVERCHARGE"][i]:
            ev.append(f"batch mass {100 * out['mass_ratio'].iloc[i]:.0f} % of nominal")
        if out["hsd_rpm_setpoint"].iloc[i] > 0 and out["hsd_kw"].iloc[i] > 0.9 * p.hsd_rated_kw:
            ev.append(f"HSD at {100 * out['hsd_kw'].iloc[i] / p.hsd_rated_kw:.0f} % load")
        if not ev:
            ev.append("no fault signature: heat load vs cooling capacity at these set-points (recipe / season)")
        causes.append("; ".join(ev))
    out["likely_cause"] = causes
    return out


def alert_table(run: pd.DataFrame) -> pd.DataFrame:
    """Collapse the row-level codes into one row per alert episode: first time raised, duration,
    ETA at the moment it was raised, cause, and whether the limit was later actually crossed."""
    rows = []
    for k in ["TEMP_LIMIT", "TEMP_PREALARM", "LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"]:
        m = run[f"rule_{k}"].values
        if not m.any():
            continue
        i = 0
        n = len(m)
        while i < n:
            if m[i]:
                j = i
                while j < n and m[j]:
                    j += 1
                first = run.iloc[i]
                actual = run["T_meas"].iloc[i:] .max() > run["limit_c"].iloc[0]
                t_cross = run.loc[(run["t_s"] >= first["t_s"]) & (run["T_meas"] > run["limit_c"]), "t_min"]
                rows.append({"batch_id": first["batch_id"], "alert": k, "raised_at_min": round(first["t_min"], 1),
                             "step": first["step"], "duration_min": round((j - i) * (run["t_s"].iloc[1] - run["t_s"].iloc[0]) / 60.0, 1),
                             "breach_eta_min": None if pd.isna(first["breach_eta_min"]) else round(first["breach_eta_min"], 1),
                             "T_now_c": round(first["T_meas"], 1), "T_next5_c": None if pd.isna(first["T_next5"]) else round(first["T_next5"], 1),
                             "likely_cause": first["likely_cause"] if k.startswith("TEMP") else _fault_evidence(k, first),
                             "limit_crossed_later": bool(actual) if k.startswith("TEMP") else None,
                             "lead_time_min": None if t_cross.empty else round(float(t_cross.iloc[0]) - first["t_min"], 1)})
                i = j
            else:
                i += 1
    return pd.DataFrame(rows)


def _fault_evidence(k: str, row: pd.Series) -> str:
    if k == "LOW_CW_FLOW":
        return f"flow {row['cw_flow']:.1f} m3/h = {100 * row['flow_ratio']:.0f} % of set-point"
    if k == "OVERCHARGE":
        return f"mass {row['batch_mass_kg']:.0f} kg = {100 * row['mass_ratio']:.0f} % of nominal"
    if k == "BLADE_WEAR":
        lag = "" if pd.isna(row.get("grind_lag", NAN)) else f"; lab grind {row['grind_lag']:+.1f} Hegman behind the physics"
        return f"HSD power {row['kw_resid_pct']:+.1f} % vs the normal level at {row['hsd_rpm']:.0f} rpm{lag}"
    if k == "JACKET_FOULING":
        return (f"implied U {row['u_drop_pct']:.0f} % below normal; product {row['T_resid_replay']:+.1f} degC above its normal gap to the "
                f"measured-power physics; flow normal ({100 * row['flow_ratio']:.0f} %)")
    return ""


def batch_verdict(run: pd.DataFrame) -> dict:
    """One-line summary of a batch after the fact: what was flagged and whether it was right."""
    codes = [c for c in ["TEMP_LIMIT", "TEMP_PREALARM", "LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"] if run[f"rule_{c}"].any()]
    breached = bool((run["T_meas"] > run["limit_c"]).any())
    first_pre = run.loc[run["rule_TEMP_PREALARM"], "t_min"]
    first_br = run.loc[run["T_meas"] > run["limit_c"], "t_min"]
    lead = float(first_br.iloc[0] - first_pre.iloc[0]) if (not first_pre.empty and not first_br.empty) else NAN
    return {"batch_id": run["batch_id"].iloc[0], "flags": "+".join(c for c in codes if not c.startswith("TEMP")) or "NONE",
            "temp_flag": "LIMIT" if "TEMP_LIMIT" in codes else ("PREALARM" if "TEMP_PREALARM" in codes else "NONE"),
            "breached": breached, "prealarm_lead_min": lead,
            "peak_T": float(run["T_meas"].max()), "ahead_mae": float((run["T_meas"] - run["T_ahead"]).abs().mean()),
            "nowcast_mae": float((run["T_meas"] - run["T_nowcast"]).abs().mean()),
            "kw_resid_pct_hsd": float(run.loc[run["step"] == "high_speed_dispersion", "kw_resid_pct"].median()),
            "kw_resid_pct_hsd_raw": float(run.loc[run["step"] == "high_speed_dispersion", "kw_resid_pct_raw"].median()),
            "T_resid_replay_hsd": float(run.loc[run["step"] == "high_speed_dispersion", "T_resid_replay"].iloc[-1]),
            "T_resid_replay_hsd_raw": float(run.loc[run["step"] == "high_speed_dispersion", "T_resid_replay_raw"].iloc[-1]),
            "u_ratio_hsd_raw": float(run.loc[run["step"] == "high_speed_dispersion", "u_ratio_raw"].median()),
            "u_drop_pct_hsd": float(run.loc[run["step"] == "high_speed_dispersion", "u_drop_pct"].median()),
            "grind_lag_last": float(run["grind_lag"].dropna().iloc[-1]) if run["grind_lag"].notna().any() else NAN,
            "flow_ratio": float(run["flow_ratio"].median()), "mass_ratio": float(run["mass_ratio"].iloc[-1]),
            "energy_kwh": float(run["energy_cum_kwh"].iloc[-1]), "kwh_per_t": float(run["specific_energy_cum_kwh_t"].iloc[-1])}
