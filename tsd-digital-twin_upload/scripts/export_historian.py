"""
export_historian.py - turn the live historian into the canonical files the rest of the twin uses.

    python scripts/export_historian.py --historian data/live/historian.sqlite --out data/live

Writes data/live/timeseries.csv and data/live/batches.csv in the Step 3 schema, so every later
step (features, models, what-if calibration, runtime baselines, alert tuning, dashboard) runs on
REAL batches with the same commands:

    TSD_DATA_DIR=data/live python scripts/check_step4.py      (and 5, 6, 7, 8)
    TSD_DATA_DIR=data/live streamlit run dashboard/app.py

The hidden-truth columns of the simulator (blade_wear, jacket_fouling_m2k_w, fault_injected ...)
are filled with 'UNKNOWN' / NaN: on the real plant they come from the maintenance log, if at all.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import kpis, schema  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historian", type=str, default=str(ROOT / "data" / "live" / "historian.sqlite"))
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "live"))
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.historian)
    rows = con.execute("SELECT batch_id, timestamp, t_s, step, data FROM timeseries ORDER BY batch_id, t_s").fetchall()
    recs = []
    for bid, ts, t_s, step, data in rows:
        d = json.loads(data)
        d.update({"batch_id": bid, "timestamp": ts, "t_s": t_s, "step": step})
        recs.append(d)
    ts_df = pd.DataFrame(recs)
    for c in schema.measured_columns():
        if c not in ts_df.columns:
            ts_df[c] = np.nan
    ts_df = ts_df[schema.measured_columns()]
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    # keep only complete batches (all four steps) with a plausible length
    keep = ts_df.groupby("batch_id")["step"].nunique() == 4
    ts_df = ts_df[ts_df["batch_id"].isin(keep[keep].index)]
    ts_df.to_csv(out / "timeseries.csv", index=False, float_format="%.3f")

    verdicts = {bid: json.loads(v) for bid, _, v in con.execute("SELECT batch_id, finished, verdict FROM batches").fetchall()}
    brows = []
    for bid, b in ts_df.groupby("batch_id", sort=False):
        fam = b["product_family"].iloc[0]
        limit = get(cfg, f"products.{fam}.temp_limit_c")
        hsd = b[b["step"] == "high_speed_dispersion"]
        lab = b["hegman"].dropna()
        v = verdicts.get(bid, {})
        e = kpis.energy_kwh(b["hsd_kw"], b["t_s"]) + kpis.energy_kwh(b["anchor_kw"], b["t_s"])
        mass = float(b["batch_mass_kg"].iloc[-1])
        brows.append({
            "batch_id": bid, "start_time": b["timestamp"].iloc[0].isoformat(timespec="seconds"), "month": b["timestamp"].iloc[0].month,
            "recipe_id": b["recipe_id"].iloc[0], "recipe_name": b["recipe_id"].iloc[0], "product_family": fam,
            "batch_mass_nominal_kg": float(v.get("nominal_mass_kg", mass)), "batch_mass_kg": mass,
            "solids_pct": float(b["solids_pct"].iloc[0]), "pvc_pct": float(b["pvc_pct"].iloc[0]), "target_hegman": float(b["target_hegman"].iloc[0]),
            "hsd_rpm_setpoint": float(hsd["hsd_rpm_setpoint"].median()) if len(hsd) else np.nan, "anchor_rpm": float(hsd["anchor_rpm"].median()) if len(hsd) else np.nan,
            "cw_flow_setpoint_m3h": float(v.get("cw_flow_setpoint_m3h", b["cw_flow_m3h"].median())),
            "T_cw_in_mean_c": float(b["T_jacket_in_c"].mean()), "T_ambient_c": float(b["T_ambient_c"].mean()), "T_raw_c": float(b["T_product_c"].iloc[0]),
            "planned_dispersion_min": float(v.get("planned_dispersion_min", len(hsd) * 10 / 60)), "dispersion_min": len(hsd) * 10 / 60,
            "n_extensions": int(max(0, round((len(hsd) * 10 / 60 - float(v.get("planned_dispersion_min", len(hsd) * 10 / 60))) / get(cfg, "simulation.operator.dispersion_extension_min")))),
            "batch_cycle_min": len(b) * 10 / 60,
            "time_to_target_hegman_lab_min": kpis.dispersion_time_min(b, float(b["target_hegman"].iloc[0])),
            "hegman_lab_final": float(lab.iloc[-1]) if len(lab) else np.nan, "hegman_reached_lab": bool(lab.ge(float(b["target_hegman"].iloc[0])).any()),
            "viscosity_ku": float(b["viscosity_ku"].dropna().iloc[-1]) if b["viscosity_ku"].notna().any() else np.nan,
            "energy_total_kwh": e, "specific_energy_kwh_per_t": e / (mass / 1000.0),
            "peak_temp_c": float(b["T_product_c"].max()), "breached": bool(b["T_product_c"].max() > limit),
            "hsd_load_peak_pct": 100.0 * float(b["hsd_kw"].max()) / get(cfg, "equipment.hsd.motor_rated_kw"),
            # no hidden truth on the real plant
            "hegman_plateau": np.nan, "grind_energy_factor": np.nan, "viscosity_factor": np.nan, "blade_wear": np.nan, "jacket_fouling_m2k_w": np.nan,
            "cw_flow_actual_frac": np.nan, "overcharge_frac": np.nan, "fault_injected": "UNKNOWN", "faults_active": "",
            "time_to_target_hegman_true_min": np.nan, "peak_temp_true_c": np.nan, "breached_true": np.nan, "overcharged_volume": np.nan,
            "motor_limited_share_pct": np.nan,
        })
    pd.DataFrame(brows).to_csv(out / "batches.csv", index=False)
    print(f"exported {ts_df['batch_id'].nunique()} complete batches / {len(ts_df):,} rows to {out}")


if __name__ == "__main__":
    main()
