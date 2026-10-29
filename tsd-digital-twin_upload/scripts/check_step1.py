"""
check_step1.py - proves the Step 1 scaffolding works before any physics is built on it.

What it does:
  1. loads config/assumptions.yaml and validates every parameter
  2. prints the assumptions register (given vs assumption vs physical)
  3. runs engineering sanity checks on the equipment basis
  4. writes the data dictionary (the twin's tag list) to data/dictionary/
  5. builds a tiny HAND-MADE batch (not physics, not real) and runs every KPI on it
  6. asserts answers we can verify by hand

Run from the project folder:   python scripts/check_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import kpis, schema  # noqa: E402
from tsd_twin.config import assumptions_only, get, load_config, parameter_table  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 200)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def ensure_folders() -> None:
    for d in ["config", "src/tsd_twin", "scripts", "dashboard/pages",
              "data/raw", "data/processed", "data/dictionary", "models", "reports", "tests"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def build_toy_batch(cfg: dict) -> pd.DataFrame:
    """A 60-minute hand-made batch at 10 s resolution. NOT a physics simulation and NOT
    real data: the numbers are chosen so the KPI answers can be checked by hand."""
    dt = get(cfg, "data.sample_period_s")
    t = np.arange(0, 3600 + dt, dt)                     # 0 .. 3600 s -> 361 rows
    n = t.size

    # step boundaries (seconds): charge 0-300, pigment 300-1200, HSD 1200-3000, letdown 3000-3600
    step = np.where(t < 300, "charge_liquids",
            np.where(t < 1200, "pigment_addition",
             np.where(t < 3000, "high_speed_dispersion", "letdown")))

    hsd_rpm = np.select([step == "charge_liquids", step == "pigment_addition",
                         step == "high_speed_dispersion"], [250, 450, 950], default=400)
    anchor_rpm = np.select([step == "charge_liquids", step == "pigment_addition",
                            step == "high_speed_dispersion"], [20, 30, 35], default=20)
    hsd_kw = np.select([step == "charge_liquids", step == "pigment_addition",
                        step == "high_speed_dispersion"], [3.0, 10.0, 45.0], default=8.0)
    anchor_kw = np.select([step == "charge_liquids", step == "pigment_addition",
                           step == "high_speed_dispersion"], [2.0, 3.0, 4.0], default=2.0)

    # product temperature: flat 28 degC, then a straight rise 28 -> 47 during HSD, then cooling
    T = np.full(n, 28.0)
    in_hsd = step == "high_speed_dispersion"
    T[in_hsd] = 28.0 + 19.0 * (t[in_hsd] - 1200) / 1800.0
    in_ld = step == "letdown"
    T[in_ld] = 47.0 - 3.0 * (t[in_ld] - 3000) / 600.0

    # jacket: inlet 18 degC, outlet rises with duty
    T_in = np.full(n, 18.0)
    T_out = T_in + np.where(in_hsd, 3.0, 1.0)
    cw_flow = np.full(n, 10.0)

    # batch mass: ramps to 3000 kg by the end of pigment addition
    mass = np.where(t < 300, 1800.0 + 400.0 * t / 300.0,
             np.where(t < 1200, 2200.0 + 800.0 * (t - 300) / 900.0, 3000.0))

    # lab Hegman: sparse readings during HSD only (target 6 reached at t = 2400 s)
    hegman = np.full(n, np.nan)
    for t_read, val in [(1800, 5.0), (2400, 6.0), (2880, 6.5)]:
        hegman[t == t_read] = val
    viscosity = np.full(n, np.nan)
    viscosity[t == 3600] = 105.0

    # a "prediction" that is deliberately 0.3 degC high with a small wobble (hand-made)
    T_pred = T + 0.3 + 0.4 * np.sin(t / 400.0)

    return pd.DataFrame({
        "batch_id": "TOY-0001", "t_s": t, "step": step,
        "product_family": "water_based", "recipe_id": "TOY-MATT",
        "batch_mass_kg": mass, "solids_pct": 58.0, "pvc_pct": 55.0, "target_hegman": 6.0,
        "hsd_rpm": hsd_rpm, "anchor_rpm": anchor_rpm, "hsd_kw": hsd_kw, "anchor_kw": anchor_kw,
        "T_product_c": T, "T_jacket_in_c": T_in, "T_jacket_out_c": T_out, "cw_flow_m3h": cw_flow,
        "T_ambient_c": 30.0, "hegman": hegman, "viscosity_ku": viscosity, "T_pred_c": T_pred,
    })


def main() -> None:
    ensure_folders()

    # ---------------------------------------------------------------- 1 + 2
    banner("1. ASSUMPTIONS FILE")
    cfg = load_config()
    table = parameter_table(cfg)
    reg_path = ROOT / "reports" / "assumptions_register.csv"
    table.to_csv(reg_path, index=False)
    print(f"Loaded {len(table)} parameters from config/assumptions.yaml - all well-formed.")
    print(table["basis"].value_counts().rename("count").to_string())
    print(f"\nFull register written to {reg_path.relative_to(ROOT)}")

    banner("2. ASSUMPTIONS TO CONFIRM (basis = assumption)")
    print(assumptions_only(cfg)[["parameter", "value", "unit", "note"]].to_string(index=False))

    # -------------------------------------------------------------------- 3
    banner("3. ENGINEERING SANITY CHECKS ON THE EQUIPMENT BASIS")
    D = get(cfg, "equipment.hsd.blade_diameter_m")
    n_max = get(cfg, "equipment.hsd.speed_max_rpm")
    v_lo, v_hi = get(cfg, "equipment.hsd.tip_speed_target_ms")
    print(f"Tip speed at VFD max {n_max} rpm      : {kpis.tip_speed_ms(n_max, D):5.1f} m/s  "
          f"(target band {v_lo}-{v_hi} m/s)")
    print(f"rpm for {v_lo} m/s / {v_hi} m/s tip speed : "
          f"{kpis.rpm_for_tip_speed(v_lo, D):4.0f} / {kpis.rpm_for_tip_speed(v_hi, D):4.0f} rpm")
    print(f"-> normal high-speed dispersion runs at ~{kpis.rpm_for_tip_speed(v_lo, D):.0f}-"
          f"{kpis.rpm_for_tip_speed(v_hi, D):.0f} rpm; 1450 rpm is drive headroom, not a working point.")

    vol_l = get(cfg, "equipment.vessel.working_volume_l")
    m_lo, m_hi = get(cfg, "batch.mass_range_t")
    print(f"\nBatch mass vs {vol_l} L working volume:")
    for fam in ("water_based", "solvent_based"):
        d_lo, d_hi = get(cfg, f"products.{fam}.density_range_kg_l")
        print(f"  {fam:14s} density {d_lo:.2f}-{d_hi:.2f} kg/L -> max batch "
              f"{vol_l * d_lo / 1000:.2f}-{vol_l * d_hi / 1000:.2f} t  (requested max {m_hi} t)")
    print(f"-> {m_hi} t needs density >= {m_hi * 1000 / vol_l:.2f} kg/L; above that mass the vessel is "
          f"overcharged (one of the Step 3 fault cases).")

    # -------------------------------------------------------------------- 4
    banner("4. DATA DICTIONARY")
    dict_path = schema.write_dictionary(ROOT / "data" / "dictionary" / "batch_timeseries_schema.csv")
    print(f"{len(schema.measured_columns())} measured columns + {len(schema.twin_columns())} twin-output "
          f"columns written to {dict_path.relative_to(ROOT)}")
    print(schema.as_dataframe()[["column", "unit", "source", "sampling"]].to_string(index=False))

    # -------------------------------------------------------------------- 5
    banner("5. KPI FUNCTIONS ON A HAND-MADE TOY BATCH  (not physics, not real data)")
    df = build_toy_batch(cfg)
    fam = "water_based"
    process = kpis.batch_process_kpis(
        df, batch_mass_kg=float(df["batch_mass_kg"].iloc[-1]),
        temp_limit_c=get(cfg, f"products.{fam}.temp_limit_c"),
        target_hegman=float(df["target_hegman"].iloc[0]),
        hsd_rated_kw=get(cfg, "equipment.hsd.motor_rated_kw"),
        anchor_rated_kw=get(cfg, "equipment.anchor.motor_rated_kw"),
        water_rho=get(cfg, "equipment.cooling_jacket.water_density_kg_m3"),
        water_cp=get(cfg, "equipment.cooling_jacket.water_cp_kj_kgk"),
    )
    print("PROCESS KPIs")
    for k, v in process.items():
        print(f"  {k:28s} {v:>10.2f}" if isinstance(v, float) else f"  {k:28s} {v!s:>10}")

    # pretend the twin predicted 10 % more energy, 22 min dispersion time, same Hegman,
    # and raised the alert 6 minutes before the actual breach
    predicted = {"energy_total_kwh": process["energy_total_kwh"] * 1.10,
                 "dispersion_time_min": 22.0, "hegman_final": 6.5, "breached": True}
    t_alert_s = process["first_breach_min"] * 60.0 - 6.0 * 60.0
    model = kpis.batch_model_kpis(df, actual=process, predicted=predicted, t_alert_s=t_alert_s)
    print("\nMODEL KPIs (toy prediction vs toy batch)")
    for k, v in model.items():
        print(f"  {k:28s} {v:>10.2f}" if isinstance(v, float) else f"  {k:28s} {v!s:>10}")

    targets = {"temp_mae_c": get(cfg, "kpi_targets.temp_mae_c"),
               "temp_max_abs_error_c": get(cfg, "kpi_targets.temp_max_abs_error_c"),
               "energy_error_pct": get(cfg, "kpi_targets.energy_error_pct"),
               "dispersion_time_error_pct": get(cfg, "kpi_targets.dispersion_time_error_pct")}
    fleet = kpis.fleet_model_kpis(pd.DataFrame([model]), targets)
    print("\nFLEET ROLL-UP (one batch, so mean = median = p90)")
    print(fleet.to_string(index=False))
    print("\nBREACH ALARM STATS:", kpis.breach_alarm_stats(pd.DataFrame([model])))
    df.to_csv(ROOT / "reports" / "step1_toy_batch.csv", index=False)

    # -------------------------------------------------------------------- 6
    banner("6. HAND-CHECKABLE ASSERTIONS")
    # 60 kW for exactly 1 hour must be 60 kWh
    e_const = kpis.energy_kwh([60.0] * 361, np.arange(0, 3601, 10))
    assert abs(e_const - 60.0) < 1e-9, e_const
    print(f"  60 kW x 1 h                          -> {e_const:.3f} kWh   OK")
    # tip speed at 1000 rpm, 0.45 m: pi*0.45*1000/60 = 23.56 m/s
    v1000 = kpis.tip_speed_ms(1000, D)
    assert abs(v1000 - 23.56) < 0.01, v1000
    print(f"  tip speed 1000 rpm, {D} m blade        -> {v1000:.2f} m/s    OK")
    # Hegman 6 first read at t = 2400 s, HSD started at 1200 s -> 20 min
    assert abs(process["dispersion_time_min"] - 20.0) < 1e-9, process["dispersion_time_min"]
    print(f"  dispersion time to Hegman 6            -> {process['dispersion_time_min']:.1f} min   OK")
    # peak 47 degC vs 45 limit: breached, exceedance 2.0 degC
    assert process["breached"] and abs(process["max_exceedance_c"] - 2.0) < 1e-9
    print(f"  limit breach (47 vs 45 degC)           -> {process['duration_min']:.1f} min above, "
          f"+{process['max_exceedance_c']:.1f} degC   OK")
    # cooling duty during HSD: 10 m3/h * 998/3600 kg/s * 4.18 * 3 K = 34.8 kW
    q_hsd = kpis.cooling_duty_kw(10.0, 18.0, 21.0, 998.0, 4.18)
    assert abs(q_hsd - 34.76) < 0.05, q_hsd
    print(f"  cooling duty 10 m3/h, dT 3 K           -> {q_hsd:.1f} kW    OK")
    # energy error we injected was exactly +10 %
    assert abs(model["energy_error_pct"] - 10.0) < 1e-9
    print(f"  injected +10 % energy error recovered  -> {model['energy_error_pct']:.1f} %     OK")
    # alert 6 min ahead of breach
    assert abs(model["breach_lead_time_min"] - 6.0) < 1e-9
    print(f"  breach lead time                       -> {model['breach_lead_time_min']:.1f} min   OK")

    banner("ALL STEP 1 CHECKS PASSED - ready for Step 2 (physics model)")


if __name__ == "__main__":
    main()
