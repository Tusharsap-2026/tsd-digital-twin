"""
features.py - data preparation and features for the twin (Step 4).

Three products, all keyed by batch_id:

  ROW FEATURES   (one row per 10 s sample)      -> data/processed/row_features.csv.gz
      for the temperature model. Alongside the measured tags they carry the PHYSICS
      BASELINES: the Step 2 model replayed for that batch with the measured
      set-points, chilled-water inlet and flow ("open" = modelled power, "replay" =
      measured power). The ML in Step 5 learns the residual (measured - physics).
      "lag_" columns are the online features: what was known HORIZON minutes earlier.

  BATCH FEATURES (one row per batch)             -> data/processed/batch_features.csv
      for the energy and time-to-Hegman models, as known at the START OF HIGH-SPEED
      DISPERSION: recipe and composition, set-points, utilities, what the charge and
      pigment phases showed (kW at a known speed = viscosity-lot indicator), the
      first 5 min of dispersion, and the physics prediction for the planned schedule.

  SPLIT          (batch_id -> train/test)        -> data/processed/split.csv
      by batch, never by row: 'split_random' (stratified by family) for model
      accuracy, 'split_time' (first 70 % of the year) for drift / fault work, and
      a 5-fold group assignment for cross-validation.

Nothing here reads timeseries_truth.csv or the fault columns: the twin only sees
what the plant will measure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import kpis, physics as ph, schema
from .config import get

NAN = float("nan")
STEP_IDX = {s: i for i, s in enumerate(schema.STEP_NAMES)}


def backward_slope(y, dt: float) -> np.ndarray:
    """Causal slope per minute: (y_i - y_{i-1}) / dt, first sample repeats the second."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    if y.size == 0:
        return out
    out[1:] = (y[1:] - y[:-1]) / dt * 60.0
    out[0] = out[1] if y.size > 1 else 0.0
    return out


# ======================================================= physics replay ==
def recipe_from_measured(b: pd.DataFrame, batch_mass_kg: float | None = None) -> ph.Recipe:
    """The recipe as the MES knows it. batch_mass_kg: the NOMINAL (recipe) mass - the causal choice,
    known before the batch starts; an overcharge then shows up as a residual (the OVERCHARGE rule).
    Falls back to the last load-cell reading when no nominal mass is given."""
    mass = float(batch_mass_kg) if batch_mass_kg is not None and not np.isnan(batch_mass_kg) else float(b["batch_mass_kg"].iloc[-1])
    return ph.Recipe(str(b["product_family"].iloc[0]), mass,
                     float(b["solids_pct"].iloc[0]), float(b["pvc_pct"].iloc[0]),
                     float(b["target_hegman"].iloc[0]), str(b["recipe_id"].iloc[0]))


def phases_from_measured(b: pd.DataFrame, dt: float, comp: dict,
                         hsd_duration_s: float | None = None,
                         letdown_duration_s: float | None = None) -> list[ph.Phase]:
    """Rebuild the four phases from the measured step column: durations from the row
    counts, set-points from the medians, dosed masses from the load-cell increments.
    hsd_duration_s / letdown_duration_s override the measured durations (planned schedule)."""
    phases = []
    mass_prev = 0.0
    for name in schema.STEP_NAMES:
        s = b[b["step"] == name]
        if s.empty:
            continue
        duration = len(s) * dt
        mass_end = float(s["batch_mass_kg"].iloc[-1])
        kind = {"charge_liquids": "liquid", "pigment_addition": "pigment",
                "high_speed_dispersion": "none", "letdown": "letdown"}[name]
        added = max(mass_end - mass_prev, 0.0) if kind != "none" else 0.0
        if name == "high_speed_dispersion" and hsd_duration_s is not None:
            duration = hsd_duration_s
        if name == "letdown":
            if letdown_duration_s is not None:
                duration = letdown_duration_s
            added = comp["m_letdown_kg"] if hsd_duration_s is not None else added
        phases.append(ph.Phase(name, duration, float(s["hsd_rpm_setpoint"].median()),
                               float(s["anchor_rpm"].median()), added, kind))
        mass_prev = mass_end
    return phases


def _interp(t: np.ndarray, y: np.ndarray):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    return lambda x: float(np.interp(x, t, y))


def conditions_from_measured(b: pd.DataFrame, replay_power: bool) -> ph.Conditions:
    """Boundary conditions as measured: first product temperature, jacket inlet and
    flow as functions of time, hall temperature; optionally the measured HSD kW."""
    t = b["t_s"].values
    T0 = float(b["T_product_c"].iloc[0])
    return ph.Conditions(T0_c=T0, T_raw_c=T0, T_cw_in_c=float(b["T_jacket_in_c"].mean()),
                         cw_flow_m3h=float(b["cw_flow_m3h"].mean()), T_ambient_c=float(b["T_ambient_c"].mean()),
                         T_cw_in_fn=_interp(t, b["T_jacket_in_c"].values),
                         cw_flow_fn=_interp(t, b["cw_flow_m3h"].values),
                         hsd_kw_fn=_interp(t, b["hsd_kw"].values) if replay_power else None,
                         hsd_rpm_fn=_interp(t, b["hsd_rpm"].values) if replay_power else None)


def physics_baselines(b: pd.DataFrame, cfg: dict, planned_dispersion_min: float | None = None,
                      batch_mass_kg: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Run the physics three ways for one batch:
      open     - modelled power, measured schedule and utilities   (what-if style)
      replay   - measured power, measured schedule and utilities   (live-twin style)
      planned  - modelled power, planned HSD time, nominal letdown (the pre-batch forecast)
    Returns row-aligned baselines and the planned-run summary."""
    dt = float(get(cfg, "data.sample_period_s"))
    recipe = recipe_from_measured(b, batch_mass_kg)
    p = ph.PhysicsParams(cfg, recipe.family)
    comp = ph.composition(recipe, p)
    phases = phases_from_measured(b, dt, comp)
    cond_open = conditions_from_measured(b, replay_power=False)
    cond_replay = conditions_from_measured(b, replay_power=True)
    df_open, _ = ph.simulate_batch(recipe, phases, cond_open, cfg)
    df_rep, _ = ph.simulate_batch(recipe, phases, cond_replay, cfg)
    n = len(b)
    out = pd.DataFrame({
        "T_phys_open": df_open["T_product_c"].values[:n],
        "T_phys_replay": df_rep["T_product_c"].values[:n],
        "hsd_kw_phys": df_open["hsd_kw"].values[:n],
        "anchor_kw_phys": df_open["anchor_kw"].values[:n],
        "hegman_phys": df_open["hegman_model"].values[:n],
        "mu_phys": df_rep["mu_pa_s"].values[:n],
        "U_phys": df_rep["U_w_m2k"].values[:n],
        "A_phys_m2": df_rep["A_wetted_m2"].values[:n],
        "Q_jacket_phys_kw": df_rep["Q_jacket_kw"].values[:n],
        "energy_phys_cum_kwh": df_open["energy_cum_kwh"].values[:n],
        "submergence_phys": df_rep["submergence"].values[:n],
        # open-run states (no measured power anywhere): what a hypothetical what-if batch can provide
        "mu_phys_open": df_open["mu_pa_s"].values[:n],
        "U_phys_open": df_open["U_w_m2k"].values[:n],
        "Q_jacket_phys_open_kw": df_open["Q_jacket_kw"].values[:n],
    })
    planned = {}
    if planned_dispersion_min is not None:
        lo, hi = get(cfg, "process_steps.letdown.duration_range_min")
        phases_pl = phases_from_measured(b, dt, comp, hsd_duration_s=60.0 * planned_dispersion_min,
                                         letdown_duration_s=60.0 * 0.5 * (lo + hi))
        hsd_rows = b[b["step"] == "high_speed_dispersion"]
        t0 = float(hsd_rows["t_s"].iloc[0]) if not hsd_rows.empty else float(b["t_s"].iloc[-1])
        known = b[b["t_s"] <= t0 + 300]                       # utilities known up to 5 min into HSD
        cond_pl = conditions_from_measured(known, replay_power=False)
        _, s = ph.simulate_batch(recipe, phases_pl, cond_pl, cfg)
        planned = {"phys_energy_total_kwh": s["energy_total_kwh"],
                   "phys_specific_energy_kwh_per_t": s["specific_energy_kwh_per_t"],
                   "phys_time_to_target_min": s["time_to_target_hegman_min"],
                   "phys_hegman_end": s["hegman_end"], "phys_peak_temp_c": s["peak_temp_c"],
                   "phys_breached": s["breached"], "phys_hsd_load_peak_pct": s["hsd_load_peak_pct"]}
    return out, planned


# ========================================================= row features ==
def row_features_for_batch(b: pd.DataFrame, base: pd.DataFrame, cfg: dict, horizon_min: float,
                           batch_mass_kg: float | None = None) -> pd.DataFrame:
    """Row features for one batch: measured tags, derived quantities, physics baselines,
    online lag features and the targets."""
    dt = float(get(cfg, "data.sample_period_s"))
    rho_w = get(cfg, "equipment.cooling_jacket.water_density_kg_m3")
    cp_w = get(cfg, "equipment.cooling_jacket.water_cp_kj_kgk")
    D = get(cfg, "equipment.hsd.blade_diameter_m")
    recipe = recipe_from_measured(b, batch_mass_kg)
    p = ph.PhysicsParams(cfg, recipe.family)
    comp = ph.composition(recipe, p)
    f = pd.DataFrame(index=b.index)
    # keys and time
    f["batch_id"] = b["batch_id"].values
    f["timestamp"] = b["timestamp"].values
    f["t_s"] = b["t_s"].values
    f["t_min"] = b["t_s"].values / 60.0
    f["step"] = b["step"].values
    f["step_idx"] = b["step"].map(STEP_IDX).values
    f["t_in_step_min"] = b.groupby("step").cumcount().values * dt / 60.0
    for s in schema.STEP_NAMES:
        f[f"is_{s}"] = (b["step"] == s).astype(int).values
    # recipe and composition (known from MES)
    f["recipe_id"] = b["recipe_id"].values
    f["family_wb"] = int(recipe.family == "water_based")
    f["solids_pct"] = recipe.solids_pct
    f["pvc_pct"] = recipe.pvc_pct
    f["target_hegman"] = recipe.target_hegman
    f["batch_mass_final_kg"] = recipe.batch_mass_kg
    f["millbase_mass_kg"] = comp["m_charge_kg"] + comp["m_pigment_kg"]
    f["phi_millbase"] = (comp["m_pigment_kg"] / p.rho_p) / (comp["m_pigment_kg"] / p.rho_p + comp["m_charge_kg"] / p.rho_l)
    f["cp_final_kj_kgk"] = comp["cp_final_kj_kgk"]
    f["density_final_kg_l"] = comp["final_density_kg_l"]
    # actuators and utilities as measured
    for c in ["hsd_rpm_setpoint", "hsd_rpm", "anchor_rpm", "hsd_kw", "anchor_kw", "batch_mass_kg",
              "T_jacket_in_c", "T_jacket_out_c", "cw_flow_m3h", "T_ambient_c"]:
        f[c] = b[c].values
    f["tip_speed_ms"] = np.pi * D * b["hsd_rpm"].values / 60.0
    f["mass_frac_of_final"] = b["batch_mass_kg"].values / recipe.batch_mass_kg
    f["cooling_duty_meas_kw"] = kpis.cooling_duty_kw(b["cw_flow_m3h"], b["T_jacket_in_c"], b["T_jacket_out_c"], rho_w, cp_w)
    # energy (from measured kW)
    kw_total = (b["hsd_kw"] + b["anchor_kw"]).values
    e_cum = np.concatenate([[0.0], np.cumsum(0.5 * (kw_total[1:] + kw_total[:-1]) * dt / 3600.0)])
    f["energy_cum_kwh"] = e_cum
    f["specific_energy_cum_kwh_t"] = e_cum / (recipe.batch_mass_kg / 1000.0)
    f["hsd_kw_per_t_millbase"] = b["hsd_kw"].values / (f["millbase_mass_kg"].values / 1000.0)
    # physics baselines and their derivatives
    for c in base.columns:
        f[c] = base[c].values
    f["dT_phys_to_jacket_c"] = base["T_phys_replay"].values - b["T_jacket_in_c"].values
    f["dTdt_phys_c_per_min"] = backward_slope(base["T_phys_replay"].values, dt)
    f["dT_phys_open_to_jacket_c"] = base["T_phys_open"].values - b["T_jacket_in_c"].values
    f["dTdt_phys_open_c_per_min"] = backward_slope(base["T_phys_open"].values, dt)
    f["hsd_kw_resid_phys"] = b["hsd_kw"].values - base["hsd_kw_phys"].values
    f["anchor_kw_resid_phys"] = b["anchor_kw"].values - base["anchor_kw_phys"].values
    # online lag features: known 'horizon' minutes before this row
    lag = int(round(horizon_min * 60.0 / dt))
    T_meas = b["T_product_c"].values
    resid_replay = T_meas - base["T_phys_replay"].values
    resid_open = T_meas - base["T_phys_open"].values
    slope = backward_slope(T_meas, dt)
    slope_2min = pd.Series(slope).rolling(int(120 / dt), min_periods=1).mean().values
    f["lag_T_meas_c"] = pd.Series(T_meas).shift(lag).values
    f["lag_resid_replay_c"] = pd.Series(resid_replay).shift(lag).values
    f["lag_resid_open_c"] = pd.Series(resid_open).shift(lag).values
    f["lag_dTdt_meas_c_per_min"] = pd.Series(slope_2min).shift(lag).values
    f["lag_T_phys_replay_c"] = pd.Series(base["T_phys_replay"].values).shift(lag).values
    # targets
    f["T_product_c"] = T_meas
    f["resid_open_c"] = resid_open
    f["resid_replay_c"] = resid_replay
    f["hegman_lab"] = b["hegman"].values
    return f


# ======================================================= batch features ==
def batch_features_for_batch(b: pd.DataFrame, f: pd.DataFrame, planned: dict, batch_row: pd.Series,
                             cfg: dict) -> dict:
    """Everything known at the start of high-speed dispersion (plus the first 5 min),
    and the outcomes to predict."""
    recipe = recipe_from_measured(b, float(batch_row["batch_mass_nominal_kg"]))
    D = get(cfg, "equipment.hsd.blade_diameter_m")
    hsd = b[b["step"] == "high_speed_dispersion"]
    pig = b[b["step"] == "pigment_addition"]
    chg = b[b["step"] == "charge_liquids"]
    t0 = float(hsd["t_s"].iloc[0])
    fh = f[f["step"] == "high_speed_dispersion"]
    first5 = hsd[hsd["t_s"] < t0 + 300]
    f5 = fh[fh["t_s"] < t0 + 300]
    fp = f[f["step"] == "pigment_addition"]
    e_at_t0 = float(f.loc[f["t_s"] == t0, "energy_cum_kwh"].iloc[0])
    e_total = float(f["energy_cum_kwh"].iloc[-1])
    lab = b["hegman"].dropna()
    row = {
        "batch_id": b["batch_id"].iloc[0], "recipe_id": recipe.recipe_id, "product_family": recipe.family,
        "start_time": b["timestamp"].iloc[0], "month": pd.Timestamp(b["timestamp"].iloc[0]).month,
        # recipe / composition
        "family_wb": int(recipe.family == "water_based"), "batch_mass_kg": recipe.batch_mass_kg,
        "batch_mass_actual_kg": float(b["batch_mass_kg"].iloc[-1]),
        "solids_pct": recipe.solids_pct, "pvc_pct": recipe.pvc_pct, "target_hegman": recipe.target_hegman,
        "millbase_mass_kg": float(f["millbase_mass_kg"].iloc[0]), "phi_millbase": float(f["phi_millbase"].iloc[0]),
        "cp_final_kj_kgk": float(f["cp_final_kj_kgk"].iloc[0]), "density_final_kg_l": float(f["density_final_kg_l"].iloc[0]),
        # set-points and utilities at t0
        "hsd_rpm_setpoint": float(hsd["hsd_rpm_setpoint"].median()),
        "tip_speed_setpoint_ms": np.pi * D * float(hsd["hsd_rpm_setpoint"].median()) / 60.0,
        "anchor_rpm_hsd": float(hsd["anchor_rpm"].median()),
        "cw_flow_m3h_at_t0": float(pig["cw_flow_m3h"].mean()),
        "T_cw_in_at_t0_c": float(pig["T_jacket_in_c"].mean()),
        "T_ambient_c": float(b["T_ambient_c"].mean()),
        "T_start_c": float(b["T_product_c"].iloc[0]),
        "T_at_hsd_start_c": float(hsd["T_product_c"].iloc[0]),
        "planned_dispersion_min": float(batch_row["planned_dispersion_min"]),
        # what the charge and pigment phases showed
        "charge_min": len(chg) * 10 / 60.0, "pigment_min": len(pig) * 10 / 60.0,
        "energy_at_hsd_start_kwh": e_at_t0,
        "pigment_kw_mean": float(pig["hsd_kw"].mean()), "pigment_rpm_mean": float(pig["hsd_rpm"].mean()),
        "pigment_kw_per_t_millbase": float(fp["hsd_kw_per_t_millbase"].mean()),
        "pigment_kw_resid_phys": float(fp["hsd_kw_resid_phys"].mean()),
        "pigment_kw_resid_phys_pct": 100.0 * float(fp["hsd_kw_resid_phys"].mean() / max(fp["hsd_kw_phys"].mean(), 0.5)),
        "pigment_anchor_kw_resid_pct": 100.0 * float(fp["anchor_kw_resid_phys"].mean() / max(fp["anchor_kw_phys"].mean(), 0.2)),
        "pigment_T_resid_replay_c": float(fp["resid_replay_c"].mean()),
        # first 5 min of dispersion
        "hsd5_kw_mean": float(first5["hsd_kw"].mean()),
        "hsd5_kw_resid_phys": float(f5["hsd_kw_resid_phys"].mean()),
        "hsd5_kw_resid_phys_pct": 100.0 * float(f5["hsd_kw_resid_phys"].mean() / max(f5["hsd_kw_phys"].mean(), 0.5)),
        "hsd5_dTdt_meas_c_per_min": 60.0 * (float(first5["T_product_c"].iloc[-1]) - float(first5["T_product_c"].iloc[0])) / max(len(first5) * 10.0, 10.0),
        "hsd5_dTdt_phys_c_per_min": float(f5["dTdt_phys_c_per_min"].mean()),
        "hsd5_T_resid_replay_c": float(f5["resid_replay_c"].mean()),
        "hsd5_rpm_sag": float(hsd["hsd_rpm_setpoint"].median() - first5["hsd_rpm"].mean()),
        "hsd5_cooling_duty_kw": float(f5["cooling_duty_meas_kw"].mean()),
        # physics forecast for the planned schedule, and physics energy for the schedule actually run
        **planned,
        "phys_energy_actual_sched_kwh": float(f["energy_phys_cum_kwh"].iloc[-1]),
        "phys_hsd_kw_mean_dispersion": float(fh["hsd_kw_phys"].mean()),
        "phys_anchor_kw_mean_dispersion": float(fh["anchor_kw_phys"].mean()),
        # context from maintenance records (not used by the default models)
        "ctx_batches_since_blade_change": int(batch_row["ctx_batches_since_blade_change"]),
        "ctx_batches_since_jacket_clean": int(batch_row["ctx_batches_since_jacket_clean"]),
        # outcomes (targets)
        "y_energy_total_kwh": e_total,
        "y_hsd_energy_kwh": float(kpis.energy_kwh(hsd["hsd_kw"], hsd["t_s"])),
        "y_specific_energy_kwh_per_t": e_total / (recipe.batch_mass_kg / 1000.0),
        "y_dispersion_min": len(hsd) * 10 / 60.0,
        "y_time_to_target_lab_min": kpis.dispersion_time_min(b, recipe.target_hegman),
        "y_hegman_reached": bool(lab.ge(recipe.target_hegman).any()),
        "y_hegman_lab_final": float(lab.iloc[-1]) if not lab.empty else NAN,
        "y_peak_temp_c": float(b["T_product_c"].max()),
        "y_breached": bool(b["T_product_c"].max() > ph.PhysicsParams(cfg, recipe.family).T_limit),
        "y_n_extensions": int(batch_row["n_extensions"]),
    }
    return row


# ================================================================ split ==
def split_batches(batches: pd.DataFrame, cfg: dict, test_frac: float = 0.25, time_train_frac: float = 0.70,
                  n_folds: int = 5) -> pd.DataFrame:
    """Train/test assignment BY BATCH. Random split stratified by family; time split on the
    calendar order; 5 group folds for cross-validation. Deterministic from the config seed."""
    rng = np.random.default_rng(int(get(cfg, "simulation.random_seed")) + 1)
    out = batches[["batch_id", "product_family", "start_time"]].copy().sort_values("start_time").reset_index(drop=True)
    out["split_random"] = "train"
    for fam, grp in out.groupby("product_family"):
        idx = grp.index.values.copy()
        rng.shuffle(idx)
        n_test = int(round(test_frac * len(idx)))
        out.loc[idx[:n_test], "split_random"] = "test"
    cut = int(round(time_train_frac * len(out)))
    out["split_time"] = np.where(np.arange(len(out)) < cut, "train", "test")
    folds = np.arange(len(out)) % n_folds
    rng.shuffle(folds)
    out["fold"] = folds
    return out


# ============================================================ pipeline ==
def feature_dictionary() -> pd.DataFrame:
    groups = {
        "keys": ["batch_id", "timestamp", "t_s", "step", "recipe_id", "product_family", "start_time", "month"],
        "time": ["t_min", "step_idx", "t_in_step_min", "is_charge_liquids", "is_pigment_addition", "is_high_speed_dispersion", "is_letdown"],
        "recipe (MES, pre-batch)": ["family_wb", "solids_pct", "pvc_pct", "target_hegman", "batch_mass_final_kg", "batch_mass_kg", "batch_mass_actual_kg",
                                    "millbase_mass_kg", "phi_millbase", "cp_final_kj_kgk", "density_final_kg_l", "planned_dispersion_min"],
        "actuators / utilities (measured)": ["hsd_rpm_setpoint", "hsd_rpm", "tip_speed_ms", "tip_speed_setpoint_ms", "anchor_rpm", "anchor_rpm_hsd",
                                             "hsd_kw", "anchor_kw", "T_jacket_in_c", "T_jacket_out_c", "cw_flow_m3h", "cw_flow_m3h_at_t0",
                                             "T_cw_in_at_t0_c", "T_ambient_c", "mass_frac_of_final", "cooling_duty_meas_kw", "T_start_c", "T_at_hsd_start_c"],
        "energy (measured)": ["energy_cum_kwh", "specific_energy_cum_kwh_t", "hsd_kw_per_t_millbase", "energy_at_hsd_start_kwh"],
        "physics baseline": ["T_phys_open", "T_phys_replay", "hsd_kw_phys", "hegman_phys", "mu_phys", "U_phys", "A_phys_m2", "Q_jacket_phys_kw",
                             "energy_phys_cum_kwh", "submergence_phys", "dT_phys_to_jacket_c", "dTdt_phys_c_per_min", "hsd_kw_resid_phys",
                             "anchor_kw_phys", "anchor_kw_resid_phys", "mu_phys_open", "U_phys_open", "Q_jacket_phys_open_kw",
                             "dT_phys_open_to_jacket_c", "dTdt_phys_open_c_per_min",
                             "phys_energy_total_kwh", "phys_specific_energy_kwh_per_t", "phys_time_to_target_min", "phys_hegman_end",
                             "phys_peak_temp_c", "phys_breached", "phys_hsd_load_peak_pct", "phys_energy_actual_sched_kwh",
                             "phys_hsd_kw_mean_dispersion", "phys_anchor_kw_mean_dispersion"],
        "pre-HSD evidence (measured by t0)": ["charge_min", "pigment_min", "pigment_kw_mean", "pigment_rpm_mean", "pigment_kw_per_t_millbase",
                                              "pigment_kw_resid_phys", "pigment_kw_resid_phys_pct", "pigment_anchor_kw_resid_pct", "pigment_T_resid_replay_c"],
        "early-HSD evidence (t0 + 5 min)": ["hsd5_kw_mean", "hsd5_kw_resid_phys", "hsd5_kw_resid_phys_pct", "hsd5_dTdt_meas_c_per_min",
                                            "hsd5_dTdt_phys_c_per_min", "hsd5_T_resid_replay_c", "hsd5_rpm_sag", "hsd5_cooling_duty_kw"],
        "online lag (horizon earlier)": ["lag_T_meas_c", "lag_resid_replay_c", "lag_resid_open_c", "lag_dTdt_meas_c_per_min", "lag_T_phys_replay_c"],
        "context (maintenance records; off by default)": ["ctx_batches_since_blade_change", "ctx_batches_since_jacket_clean"],
        "targets": ["T_product_c", "resid_open_c", "resid_replay_c", "hegman_lab", "y_energy_total_kwh", "y_hsd_energy_kwh",
                    "y_specific_energy_kwh_per_t", "y_dispersion_min", "y_time_to_target_lab_min", "y_hegman_reached",
                    "y_hegman_lab_final", "y_peak_temp_c", "y_breached", "y_n_extensions"],
    }
    rows = [{"feature": f, "group": g} for g, fs in groups.items() for f in fs]
    return pd.DataFrame(rows)


def build_features(cfg: dict, raw_dir: Path | str, out_dir: Path | str, verbose: bool = True) -> dict:
    """Read data/raw, build everything, write data/processed. Returns the three frames."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.read_csv(raw_dir / "timeseries.csv", parse_dates=["timestamp"])
    batches = pd.read_csv(raw_dir / "batches.csv", parse_dates=["start_time"]).sort_values("start_time").reset_index(drop=True)
    horizon = float(get(cfg, "kpi_targets.breach_lead_time_min"))
    life = int(get(cfg, "simulation.faults.blade_life_batches"))
    clean = int(get(cfg, "simulation.faults.fouling_cleaning_interval_batches"))
    # maintenance counters the plant would have in its CMMS (the simulator's schedule is known)
    batches["ctx_batches_since_blade_change"] = np.arange(len(batches)) % life
    batches["ctx_batches_since_jacket_clean"] = np.arange(len(batches)) % clean
    batches = batches.set_index("batch_id")

    rows, brows = [], []
    for k, (bid, b) in enumerate(ts.groupby("batch_id", sort=False)):
        b = b.reset_index(drop=True)
        nominal = float(batches.loc[bid, "batch_mass_nominal_kg"])
        base, planned = physics_baselines(b, cfg, planned_dispersion_min=float(batches.loc[bid, "planned_dispersion_min"]), batch_mass_kg=nominal)
        f = row_features_for_batch(b, base, cfg, horizon, batch_mass_kg=nominal)
        rows.append(f)
        brows.append(batch_features_for_batch(b, f, planned, batches.loc[bid], cfg))
        if verbose and (k + 1) % 50 == 0:
            print(f"  {k + 1} batches featurised")
    row_feat = pd.concat(rows, ignore_index=True)
    batch_feat = pd.DataFrame(brows)
    split = split_batches(batches.reset_index(), cfg)
    row_feat.to_csv(out_dir / "row_features.csv.gz", index=False, float_format="%.4f", compression="gzip")
    batch_feat.to_csv(out_dir / "batch_features.csv", index=False, float_format="%.4f")
    split.to_csv(out_dir / "split.csv", index=False)
    feature_dictionary().to_csv(out_dir / "feature_dictionary.csv", index=False)
    return {"row_features": row_feat, "batch_features": batch_feat, "split": split}


def load_processed(out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    return {"row_features": pd.read_csv(out_dir / "row_features.csv.gz", parse_dates=["timestamp"]),
            "batch_features": pd.read_csv(out_dir / "batch_features.csv", parse_dates=["start_time"]),
            "split": pd.read_csv(out_dir / "split.csv", parse_dates=["start_time"])}
