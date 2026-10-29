"""
kpis.py - every KPI formula for the TSD digital twin, in one place.

Two layers of KPIs:

  PROCESS KPIs : what the batch did (energy, time, temperature, grind, utilities).
                 Computed from measured OR simulated time series.
  MODEL KPIs   : how well the twin predicted it (errors, lead time, alarms).
                 Computed by comparing predictions with measurements.

All functions use the canonical column names from schema.py, so they run
unchanged on simulated data now and on historian data later.

Equations (plain form):
  Tip speed          v   = pi * D * N / 60                          [m/s], N in rpm
  Energy             E   = integral(P dt) / 3600                    [kWh], P in kW, t in s
  Specific energy    e   = E / (m_batch / 1000)                     [kWh/t]
  Cooling duty       Q   = (V_dot * rho / 3600) * cp * (T_out - T_in)   [kW]
  Motor load         L   = 100 * P / P_rated                        [%]
  Dispersion time    t_d = t(first lab Hegman >= target) - t(HSD step start)   [min]
  Temperature MAE    MAE = mean(|T_pred - T_meas|)                  [degC]
  Percent error      err = 100 * (pred - actual) / actual           [%]
  Breach lead time   lead = t_breach - t_alert                      [min]
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# numpy renamed trapz -> trapezoid in 2.0; support both
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

HSD_STEP = "high_speed_dispersion"
NAN = float("nan")


# ------------------------------------------------------------------ basics --
def tip_speed_ms(rpm: float, blade_diameter_m: float) -> float:
    """Blade tip speed v = pi * D * N / 60."""
    return math.pi * blade_diameter_m * rpm / 60.0


def rpm_for_tip_speed(tip_speed_ms: float, blade_diameter_m: float) -> float:
    """Inverse: rpm needed for a given tip speed."""
    return tip_speed_ms * 60.0 / (math.pi * blade_diameter_m)


def energy_kwh(kw, t_s) -> float:
    """Integrate power [kW] over time [s] -> kWh (trapezoid rule, NaN rows skipped)."""
    kw = np.asarray(kw, dtype=float)
    t = np.asarray(t_s, dtype=float)
    ok = ~np.isnan(kw) & ~np.isnan(t)
    if ok.sum() < 2:
        return NAN
    return float(_trapz(kw[ok], t[ok]) / 3600.0)


def specific_energy_kwh_per_t(e_kwh: float, batch_mass_kg: float) -> float:
    """kWh per tonne of batch."""
    if batch_mass_kg is None or batch_mass_kg <= 0 or np.isnan(e_kwh):
        return NAN
    return e_kwh / (batch_mass_kg / 1000.0)


def cooling_duty_kw(cw_flow_m3h, t_in_c, t_out_c, rho_kg_m3: float = 998.0,
                    cp_kj_kgk: float = 4.18) -> np.ndarray:
    """Heat removed by the jacket [kW]; positive = cooling the product."""
    m_dot = np.asarray(cw_flow_m3h, dtype=float) * rho_kg_m3 / 3600.0      # kg/s
    d_t = np.asarray(t_out_c, dtype=float) - np.asarray(t_in_c, dtype=float)
    return m_dot * cp_kj_kgk * d_t


def motor_load_pct(kw, rated_kw: float) -> dict:
    """Mean and peak electrical load as % of the motor rating."""
    kw = np.asarray(kw, dtype=float)
    kw = kw[~np.isnan(kw)]
    if kw.size == 0 or rated_kw <= 0:
        return {"mean_pct": NAN, "peak_pct": NAN}
    return {"mean_pct": float(100.0 * kw.mean() / rated_kw),
            "peak_pct": float(100.0 * kw.max() / rated_kw)}


def pct_error(pred: float, actual: float) -> float:
    """Signed percent error of a prediction. NaN when the actual is missing or zero."""
    if actual is None or pred is None or np.isnan(actual) or np.isnan(pred) or actual == 0:
        return NAN
    return 100.0 * (pred - actual) / actual


# ------------------------------------------------------------- temperature --
def temperature_errors(t_meas, t_pred) -> dict:
    """MAE, RMSE, bias (pred - meas) and worst-point error over aligned series."""
    a = np.asarray(t_meas, dtype=float)
    p = np.asarray(t_pred, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(p)
    if ok.sum() == 0:
        return {"mae_c": NAN, "rmse_c": NAN, "bias_c": NAN, "max_abs_error_c": NAN, "n_points": 0}
    err = p[ok] - a[ok]
    return {"mae_c": float(np.mean(np.abs(err))),
            "rmse_c": float(np.sqrt(np.mean(err ** 2))),
            "bias_c": float(np.mean(err)),
            "max_abs_error_c": float(np.max(np.abs(err))),
            "n_points": int(ok.sum())}


def limit_breach(t_product_c, t_s, limit_c: float) -> dict:
    """Did the product cross the limit? For how long, by how much, and when first?"""
    T = np.asarray(t_product_c, dtype=float)
    t = np.asarray(t_s, dtype=float)
    ok = ~np.isnan(T) & ~np.isnan(t)
    T, t = T[ok], t[ok]
    if T.size == 0:
        return {"breached": False, "duration_min": NAN, "max_exceedance_c": NAN,
                "first_breach_min": NAN, "peak_temp_c": NAN, "margin_to_limit_c": NAN}
    dt = float(np.median(np.diff(t))) if T.size > 1 else 0.0
    over = T > limit_c
    return {"breached": bool(over.any()),
            "duration_min": float(over.sum() * dt / 60.0),
            "max_exceedance_c": float(max(0.0, (T - limit_c).max())),
            "first_breach_min": float(t[over][0] / 60.0) if over.any() else NAN,
            "peak_temp_c": float(T.max()),
            "margin_to_limit_c": float(limit_c - T.max())}


def lead_time_min(t_alert_s: float, t_breach_s: float) -> float:
    """Minutes of warning the twin gave before the actual breach (negative = too late)."""
    if t_alert_s is None or t_breach_s is None or np.isnan(t_alert_s) or np.isnan(t_breach_s):
        return NAN
    return (t_breach_s - t_alert_s) / 60.0


# ------------------------------------------------------------------- grind --
def dispersion_time_min(df: pd.DataFrame, target_hegman: float, hsd_step: str = HSD_STEP) -> float:
    """Minutes from the start of the high-speed step to the first Hegman reading >= target.
    Uses the lab/gauge column 'hegman' (NaN where no reading). NaN if never reached."""
    in_step = df["step"] == hsd_step
    if not in_step.any():
        return NAN
    t0 = float(df.loc[in_step, "t_s"].iloc[0])
    hit = df[(df["t_s"] >= t0) & df["hegman"].notna() & (df["hegman"] >= target_hegman)]
    if hit.empty:
        return NAN
    return (float(hit["t_s"].iloc[0]) - t0) / 60.0


def hegman_summary(df: pd.DataFrame, target_hegman: float) -> dict:
    """Final reading, whether the target was reached, how many readings were taken."""
    readings = df["hegman"].dropna()
    return {"hegman_final": float(readings.iloc[-1]) if not readings.empty else NAN,
            "hegman_reached": bool(readings.ge(target_hegman).any()),
            "n_hegman_readings": int(readings.size)}


def step_durations_min(df: pd.DataFrame) -> dict:
    """Duration of each phase and the whole batch cycle, in minutes."""
    out: dict = {}
    if df.empty:
        return out
    dt = float(np.median(np.diff(df["t_s"]))) if len(df) > 1 else 0.0
    for name, grp in df.groupby("step", sort=False):
        out[f"dur_{name}_min"] = float(len(grp) * dt / 60.0)
    out["batch_cycle_min"] = float((df["t_s"].iloc[-1] - df["t_s"].iloc[0] + dt) / 60.0)
    return out


# -------------------------------------------------------------- aggregates --
def batch_process_kpis(df: pd.DataFrame, batch_mass_kg: float, temp_limit_c: float,
                       target_hegman: float, hsd_rated_kw: float, anchor_rated_kw: float,
                       water_rho: float = 998.0, water_cp: float = 4.18) -> dict:
    """All PROCESS KPIs for one batch (measured or simulated)."""
    e_hsd = energy_kwh(df["hsd_kw"], df["t_s"])
    e_anchor = energy_kwh(df["anchor_kw"], df["t_s"])
    e_total = e_hsd + e_anchor
    q = cooling_duty_kw(df["cw_flow_m3h"], df["T_jacket_in_c"], df["T_jacket_out_c"], water_rho, water_cp)
    load_hsd = motor_load_pct(df["hsd_kw"], hsd_rated_kw)
    load_anchor = motor_load_pct(df["anchor_kw"], anchor_rated_kw)

    out = {
        "batch_mass_kg": float(batch_mass_kg),
        "energy_hsd_kwh": e_hsd,
        "energy_anchor_kwh": e_anchor,
        "energy_total_kwh": e_total,
        "specific_energy_kwh_per_t": specific_energy_kwh_per_t(e_total, batch_mass_kg),
        "dispersion_time_min": dispersion_time_min(df, target_hegman),
        "cooling_duty_mean_kw": float(np.nanmean(q)),
        "cooling_duty_peak_kw": float(np.nanmax(q)),
        "hsd_load_mean_pct": load_hsd["mean_pct"],
        "hsd_load_peak_pct": load_hsd["peak_pct"],
        "anchor_load_peak_pct": load_anchor["peak_pct"],
    }
    out.update(limit_breach(df["T_product_c"], df["t_s"], temp_limit_c))
    out.update(hegman_summary(df, target_hegman))
    out.update(step_durations_min(df))
    cycle = out.get("batch_cycle_min", NAN)
    out["throughput_t_per_h"] = (batch_mass_kg / 1000.0) / (cycle / 60.0) if cycle and cycle > 0 else NAN
    return out


def batch_model_kpis(df: pd.DataFrame, actual: dict, predicted: dict,
                     t_alert_s: float = NAN) -> dict:
    """MODEL KPIs for one batch: how far the twin's predictions were from what happened.
    df carries T_product_c and T_pred_c; `actual` and `predicted` are batch-level
    dicts with energy_total_kwh, dispersion_time_min, hegman_final, breached, first_breach_min."""
    terr = temperature_errors(df["T_product_c"], df["T_pred_c"])
    out = {f"temp_{k}": v for k, v in terr.items()}
    out["energy_error_pct"] = pct_error(predicted.get("energy_total_kwh", NAN),
                                        actual.get("energy_total_kwh", NAN))
    out["dispersion_time_error_pct"] = pct_error(predicted.get("dispersion_time_min", NAN),
                                                 actual.get("dispersion_time_min", NAN))
    out["hegman_error"] = predicted.get("hegman_final", NAN) - actual.get("hegman_final", NAN)

    actual_breach = bool(actual.get("breached", False))
    pred_breach = bool(predicted.get("breached", False))
    if actual_breach and pred_breach:
        out["breach_outcome"] = "hit"
    elif actual_breach and not pred_breach:
        out["breach_outcome"] = "miss"
    elif pred_breach and not actual_breach:
        out["breach_outcome"] = "false_alarm"
    else:
        out["breach_outcome"] = "true_negative"
    t_breach_s = actual.get("first_breach_min", NAN) * 60.0 if actual_breach else NAN
    out["breach_lead_time_min"] = lead_time_min(t_alert_s, t_breach_s) if actual_breach else NAN
    return out


def fleet_model_kpis(per_batch: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """Roll per-batch model KPIs into the honest summary: mean, median, P90, worst,
    and the share of batches that meet the target. `targets` keys match KPI names."""
    rows = []
    for col in ["temp_mae_c", "temp_max_abs_error_c", "energy_error_pct", "dispersion_time_error_pct"]:
        if col not in per_batch.columns:
            continue
        v = per_batch[col].abs().dropna()
        if v.empty:
            continue
        tgt = float(targets.get(col, NAN))
        rows.append({"kpi": col, "n_batches": int(v.size), "mean": float(v.mean()),
                     "median": float(v.median()), "p90": float(v.quantile(0.90)),
                     "worst": float(v.max()), "target": tgt,
                     "pass_rate_pct": float(100.0 * (v <= tgt).mean()) if not np.isnan(tgt) else NAN})
    return pd.DataFrame(rows)


def breach_alarm_stats(per_batch: pd.DataFrame) -> dict:
    """Hit / miss / false-alarm counts for the limit-breach warning across batches."""
    if "breach_outcome" not in per_batch.columns or per_batch.empty:
        return {}
    c = per_batch["breach_outcome"].value_counts()
    hits, misses = int(c.get("hit", 0)), int(c.get("miss", 0))
    fa, tn = int(c.get("false_alarm", 0)), int(c.get("true_negative", 0))
    n = hits + misses + fa + tn
    lead = per_batch["breach_lead_time_min"].dropna() if "breach_lead_time_min" in per_batch else pd.Series(dtype=float)
    return {"n_batches": n, "hits": hits, "misses": misses, "false_alarms": fa, "true_negatives": tn,
            "detection_rate_pct": 100.0 * hits / (hits + misses) if hits + misses else NAN,
            "false_alarms_per_100_batches": 100.0 * fa / n if n else NAN,
            "median_lead_time_min": float(lead.median()) if not lead.empty else NAN}
