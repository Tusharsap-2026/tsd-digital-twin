"""
recalibrate.py - refit the physics on measured batches, in the order that matters (Step 9).

    python scripts/recalibrate.py --data data/live --out config/assumptions.yaml        # the real plant
    python scripts/recalibrate.py --data data/raw  --out /tmp/recal.yaml --batches 120  # the test in check_step9

Input: canonical batch files (timeseries.csv + batches.csv in the Step 3 schema; the historian
exports them in the same form). Fits, in this order, because each one changes what the next
one sees:
  1. HSD power number    - measured HSD shaft power / physics shaft power at the actual speed
  2. product-side film   - the implied overall U from the measured jacket duty (upper quartile
                           of the batches = clean jacket) -> h_product
  3. evaporative loss    - the temperature residual against the measured-power physics during
                           high-speed dispersion, after 1 and 2 -> k_evap (bisection)
  4. grind energy        - lab time to target (de-biased by half a check interval) / physics time
                           -> characteristic specific energy, per family
  Not fitted: phi_max and the tip-speed effectiveness exponent - they are not identifiable from
  power and temperature alone; they need deliberate rpm / solids trials (docs/PATH_TO_LIVE.md).

Writes the new values into the physics section of the config (values AND a note with the fitted
multiplier and the batch count), then tells you which steps to re-run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import features as ft, physics as ph  # noqa: E402
from tsd_twin.config import get, load_config, update_values  # noqa: E402


def shaft_kw(elec_kw: np.ndarray, p: ph.PhysicsParams) -> np.ndarray:
    return np.maximum(elec_kw - p.hsd_no_load_kw, 0.0) * p.hsd_eff


def fit_power(batches: list[pd.DataFrame], cfg: dict) -> tuple[float, int]:
    """Median ratio of measured to physics HSD shaft power in high-speed dispersion."""
    ratios = []
    for b in batches:
        recipe = ft.recipe_from_measured(b)
        p = ph.PhysicsParams(cfg, recipe.family)
        base, _ = ft.physics_baselines(b, cfg)
        hsd = (b["step"] == "high_speed_dispersion").values
        if hsd.sum() < 30:
            continue
        r = shaft_kw(b["hsd_kw"].values[hsd], p) / np.maximum(shaft_kw(base["hsd_kw_phys"].values[hsd], p), 0.5)
        ratios.append(float(np.median(r[np.isfinite(r)])))
    return float(np.median(ratios)), len(ratios)


def fit_h_product(batches: list[pd.DataFrame], cfg: dict) -> tuple[float, int]:
    """Implied UA from the measured jacket duty -> h_product multiplier (upper quartile = clean jacket)."""
    rho_w, cp_w = get(cfg, "equipment.cooling_jacket.water_density_kg_m3"), get(cfg, "equipment.cooling_jacket.water_cp_kj_kgk")
    mults = []
    for b in batches:
        recipe = ft.recipe_from_measured(b)
        p = ph.PhysicsParams(cfg, recipe.family)
        base, _ = ft.physics_baselines(b, cfg)
        hsd = (b["step"] == "high_speed_dispersion").values
        C = b["cw_flow_m3h"].values * rho_w / 3600.0 * cp_w * 1000.0
        Q = C * (b["T_jacket_out_c"].values - b["T_jacket_in_c"].values)
        dT = b["T_product_c"].values - b["T_jacket_in_c"].values
        ok = hsd & (dT >= 5.0)
        if ok.sum() < 30:
            continue
        ua_impl = Q[ok] / np.maximum(dT[ok] - Q[ok] / (2.0 * C[ok]), 0.5)
        A = base["A_phys_m2"].values[ok]
        u_impl = np.median(ua_impl / A)
        # the twin's own h_jacket and wall terms for these conditions
        flow = float(b["cw_flow_m3h"].values[ok].mean())
        h_j = p.h_j_ref * (max(flow, 0.1) / p.flow_ref) ** p.flow_exp
        r_rest = p.wall_m / p.k_wall + 1.0 / h_j
        if 1.0 / u_impl - r_rest <= 0:
            continue
        h_p_impl = 1.0 / (1.0 / u_impl - r_rest)
        # the twin's h_product under the same anchor speed / viscosity
        h_p_model = np.median(1.0 / (1.0 / base["U_phys_open"].values[ok] - r_rest))
        mults.append(float(h_p_impl / h_p_model))
    mults = np.array(mults)
    return float(np.quantile(mults, 0.75)), len(mults)


def temp_residual(batches: list[pd.DataFrame], cfg: dict) -> float:
    """Median (measured - measured-power physics) temperature in high-speed dispersion."""
    res = []
    for b in batches:
        base, _ = ft.physics_baselines(b, cfg)
        hsd = (b["step"] == "high_speed_dispersion").values
        res.append(float(np.median((b["T_product_c"].values - base["T_phys_replay"].values)[hsd])))
    return float(np.median(res))


def fit_evaporation(batches: list[pd.DataFrame], cfg_path: Path, lo: float = 0.2, hi: float = 5.0, iters: int = 10) -> tuple[float, float]:
    """Bisection on the evaporation multiplier until the median HSD temperature residual is ~0."""
    base_vals = {fam: get(load_config(cfg_path), f"physics.losses.evaporative_loss_w_per_k.{fam}") for fam in ("water_based", "solvent_based")}

    def resid_for(mult: float) -> float:
        tmp = cfg_path.with_name("_recal_tmp.yaml")
        shutil.copy(cfg_path, tmp)
        update_values(tmp, {f"physics.losses.evaporative_loss_w_per_k.{fam}": (base_vals[fam] * mult, None) for fam in base_vals})
        r = temp_residual(batches, load_config(tmp))
        tmp.unlink()
        return r
    # more evaporation -> cooler physics -> residual rises; residual(mult) is increasing in mult
    r_lo, r_hi = resid_for(lo), resid_for(hi)
    if r_lo > 0:            # even the least evaporation leaves the physics too cool: other losses are too high
        return lo, r_lo
    if r_hi < 0:
        return hi, r_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r_mid = resid_for(mid)
        if r_mid < 0:
            lo = mid
        else:
            hi = mid
    mult = 0.5 * (lo + hi)
    return mult, resid_for(mult)


def fit_grind(batches: list[pd.DataFrame], meta: pd.DataFrame, cfg: dict) -> dict[str, tuple[float, int]]:
    """Lab time to target (de-biased by half a check interval) / physics time to target, per family."""
    every = get(cfg, "data.lab_hegman_interval_min")
    out = {}
    for fam in ("water_based", "solvent_based"):
        ratios = []
        for b in batches:
            if b["product_family"].iloc[0] != fam:
                continue
            recipe = ft.recipe_from_measured(b)
            lab_t = ft.kpis.dispersion_time_min(b, recipe.target_hegman)
            if np.isnan(lab_t):
                continue
            _, planned = ft.physics_baselines(b, cfg, planned_dispersion_min=float(meta.loc[b["batch_id"].iloc[0], "planned_dispersion_min"]))
            pt = planned.get("phys_time_to_target_min", np.nan)
            if pt is None or np.isnan(pt) or pt <= 0:
                continue
            ratios.append((lab_t - every / 2) / pt)
        out[fam] = (float(np.median(ratios)) if ratios else 1.0, len(ratios))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "live"), help="folder with timeseries.csv + batches.csv (canonical schema)")
    ap.add_argument("--out", type=str, default=str(ROOT / "config" / "assumptions.yaml"), help="config file to write (default: in place)")
    ap.add_argument("--batches", type=int, default=80, help="use the first N batches (speed)")
    a = ap.parse_args()
    data = Path(a.data)
    out = Path(a.out)
    src_cfg = ROOT / "config" / "assumptions.yaml"
    if out != src_cfg:
        shutil.copy(src_cfg, out)
    ts = pd.read_csv(data / "timeseries.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(data / "batches.csv").set_index("batch_id")
    ids = list(dict.fromkeys(ts["batch_id"]))[: a.batches]
    batches = [ts[ts["batch_id"] == bid].reset_index(drop=True) for bid in ids]
    print(f"recalibration on {len(batches)} batches from {data}  ->  {out}")

    # 1. power
    cfg = load_config(out)
    r_p, n = fit_power(batches, cfg)
    np_old = get(cfg, "physics.hsd_power.np_turbulent")
    kp = get(cfg, "physics.hsd_power.laminar_constant_kp")
    # P ~ Kp/Re + Np_turb; scale Np_turb so the total scales by r_p at a typical Re (~5000)
    np_new = r_p * (kp / 5000.0 + np_old) - kp / 5000.0
    update_values(out, {"physics.hsd_power.np_turbulent": (round(np_new, 4), f"RECALIBRATED from {n} measured batches: measured/physics HSD shaft power = {r_p:.3f} (was {np_old})")})
    print(f"  1. HSD power: measured / physics = {r_p:.3f}  ->  Np_turb {np_old} -> {np_new:.4f}   ({n} batches)")

    # 2. h_product
    cfg = load_config(out)
    m_h, n = fit_h_product(batches, cfg)
    h_old = get(cfg, "physics.heat_transfer.h_product_ref_w_m2k")
    update_values(out, {"physics.heat_transfer.h_product_ref_w_m2k": (round(h_old * m_h, 1), f"RECALIBRATED from {n} batches: implied U (clean-jacket quartile) gives h_product x {m_h:.3f} (was {h_old})")})
    print(f"  2. product-side film: multiplier {m_h:.3f}  ->  h_product {h_old} -> {h_old * m_h:.0f} W/m2K   ({n} batches)")

    # 3. evaporation
    sample = batches[: min(40, len(batches))]
    r_before = temp_residual(sample, load_config(out))
    m_e, r_after = fit_evaporation(sample, out)
    cfg = load_config(out)
    for fam in ("water_based", "solvent_based"):
        k_old = get(cfg, f"physics.losses.evaporative_loss_w_per_k.{fam}")
        update_values(out, {f"physics.losses.evaporative_loss_w_per_k.{fam}": (round(k_old * m_e, 1), f"RECALIBRATED: evaporation x {m_e:.3f} zeroes the HSD temperature residual ({r_before:+.2f} -> {r_after:+.2f} degC, {len(sample)} batches)")})
    print(f"  3. evaporation: multiplier {m_e:.3f}; HSD temperature residual {r_before:+.2f} -> {r_after:+.2f} degC   ({len(sample)} batches)")

    # 4. grind
    cfg = load_config(out)
    g = fit_grind(batches, meta, cfg)
    for fam, (m_g, n) in g.items():
        e_old = get(cfg, f"physics.grind.char_specific_energy_kwh_t.{fam}")
        update_values(out, {f"physics.grind.char_specific_energy_kwh_t.{fam}": (round(e_old * m_g, 3), f"RECALIBRATED from {n} batches: lab time to target / physics = {m_g:.3f} (was {e_old})")})
        print(f"  4. grind {fam}: lab/physics time {m_g:.3f}  ->  e_c {e_old} -> {e_old * m_g:.2f} kWh/t   ({n} batches)")

    # verification: physics-only temperature error before/after on the SAME batches
    before, after = load_config(src_cfg), load_config(out)
    maes = {"before": [], "after": []}
    for b in sample:
        for k, c in (("before", before), ("after", after)):
            base, _ = ft.physics_baselines(b, c)
            maes[k].append(float(np.mean(np.abs(b["T_product_c"].values - base["T_phys_open"].values))))
    print(f"  physics-only temperature MAE on these batches: {np.median(maes['before']):.2f} -> {np.median(maes['after']):.2f} degC (median)")
    print("\n  multipliers:", {"Np_turb": round(np_new / np_old, 3), "h_product": round(m_h, 3), "evaporation": round(m_e, 3),
                             "e_c_water": round(g['water_based'][0], 3), "e_c_solvent": round(g['solvent_based'][0], 3)})
    print(f"\n  written to {out}. Now re-run, in order: check_step4 (features), check_step5 (models), check_step6 (what-if calibration),\n"
          f"  check_step7 (runtime baselines), check_step8 (alert thresholds) - the pipeline is scripted; about 10 minutes.")


if __name__ == "__main__":
    main()
