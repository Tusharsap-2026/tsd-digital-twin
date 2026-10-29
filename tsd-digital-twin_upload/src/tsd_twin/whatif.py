"""
whatif.py - the what-if simulator and the kWh/t optimiser (Step 6).

    settings = Settings(family="water_based", batch_mass_kg=3000, solids_pct=58, pvc_pct=55,
                        target_hegman=6.0, hsd_rpm=950, anchor_rpm=35, dispersion_min=35,
                        cw_inlet_c=22, cw_flow_m3h=10, T_start_c=30, T_ambient_c=32)
    result = run_scenario(settings, cfg, twin)       # twin = load_twin(models_dir)
    result.summary -> dict (kWh, kWh/t, peak T, Hegman, batch time, pass/fail ...)
    result.curve   -> DataFrame (t_min, step, T_phys, T_pred, hegman, hsd_kw, energy_cum_kwh ...)

    best = optimise(settings, cfg, twin)              # lowest kWh/t that still passes every check

How a scenario is evaluated (two physics runs, one residual model):
  1. physics with the dispersion phase held at its maximum -> the grind curve; the CALIBRATED
     time to target decides the schedule the way the line runs it (lab checks, extensions),
  2. physics again for that schedule -> the temperature, power and energy curves,
  3. the Step 5 hybrid-forecast residual model corrects the temperature curve,
  4. energy is scaled by the calibrated plant/physics energy ratio,
  5. every limit is checked: product temperature (limit and pre-alarm), grind target reached,
     motor load, drive limit (speed sag), vessel volume, tip-speed band, blade submergence.

The two scalar calibrations (energy ratio, grind-time factor) are learned from measured
batches in check_step6.py and stored in models/whatif_calibration.json. They are the
first two numbers to refit on real batches (Step 9).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as ft, models as md, physics as ph
from .config import get

NAN = float("nan")
STEP_IDX = {s: i for i, s in enumerate(ph.STEP_ORDER)}


# =============================================================== inputs ==
@dataclass
class Settings:
    family: str                          # "water_based" | "solvent_based"
    batch_mass_kg: float
    solids_pct: float
    pvc_pct: float
    target_hegman: float
    hsd_rpm: float                       # high-speed dispersion set-point
    anchor_rpm: float                    # anchor speed during dispersion
    dispersion_min: float                # planned high-speed time (before any extension)
    cw_inlet_c: float                    # chilled-water inlet temperature
    cw_flow_m3h: float
    T_start_c: float = 28.0              # raw-material / first-charge temperature
    T_ambient_c: float = 30.0
    recipe_id: str = "WHATIF"
    hegman_plateau: float | None = None  # best achievable grind for this product (None -> family default)
    stop_at_first_pass: bool = False     # True: end dispersion at the first lab check that passes (even before plan)
    blade_wear: float = 0.0              # 0 = new blade
    jacket_fouling_m2k_w: float = 0.0    # extra jacket resistance

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Twin:
    """Everything the what-if needs besides the config: the residual model and the two calibrations."""
    temperature: md.TemperatureModels | None = None
    energy_factor: float = 1.0           # measured / physics energy, median over calibration batches
    grind_time_factor: float = 1.0       # (lab time to target) / (physics time to target), median
    n_calibration_batches: int = 0
    source: str = "none"


def load_twin(models_dir: Path | str) -> Twin:
    """Load the Step 5 temperature models and the Step 6 calibration (falls back gracefully)."""
    models_dir = Path(models_dir)
    twin = Twin()
    f = models_dir / "temperature_models.joblib"
    if f.exists():
        import joblib
        twin.temperature = joblib.load(f)
        twin.source = "models/temperature_models.joblib"
    c = models_dir / "whatif_calibration.json"
    if c.exists():
        cal = json.loads(c.read_text())
        twin.energy_factor = float(cal.get("energy_factor", 1.0))
        twin.grind_time_factor = float(cal.get("grind_time_factor", 1.0))
        twin.n_calibration_batches = int(cal.get("n_batches", 0))
    return twin


# ========================================================= the scenario ==
@dataclass
class ScenarioResult:
    settings: Settings
    summary: dict
    curve: pd.DataFrame
    checks: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _phases(s: Settings, cfg: dict, dispersion_min: float) -> list[ph.Phase]:
    recipe = _recipe(s)
    phases = ph.standard_phases(recipe, cfg, dispersion_rpm=s.hsd_rpm, dispersion_min=dispersion_min)
    phases[2].anchor_rpm = float(s.anchor_rpm)
    return phases


def _recipe(s: Settings) -> ph.Recipe:
    return ph.Recipe(s.family, float(s.batch_mass_kg), float(s.solids_pct), float(s.pvc_pct),
                     float(s.target_hegman), s.recipe_id, hegman_plateau=s.hegman_plateau)


def _conditions(s: Settings) -> ph.Conditions:
    return ph.Conditions(T0_c=s.T_start_c, T_raw_c=s.T_start_c, T_cw_in_c=s.cw_inlet_c,
                         cw_flow_m3h=s.cw_flow_m3h, T_ambient_c=s.T_ambient_c)


def _forecast_features(df: pd.DataFrame, recipe: ph.Recipe, s: Settings, cfg: dict) -> pd.DataFrame:
    """The FEATURES_T_FORECAST columns for a hypothetical batch, from one open physics run."""
    dt = float(get(cfg, "data.sample_period_s"))
    p = ph.PhysicsParams(cfg, recipe.family)
    comp = ph.composition(recipe, p)
    f = pd.DataFrame(index=df.index)
    f["T_phys_open"] = df["T_product_c"].values
    f["dT_phys_open_to_jacket_c"] = df["T_product_c"].values - df["T_jacket_in_c"].values
    f["dTdt_phys_open_c_per_min"] = ft.backward_slope(df["T_product_c"].values, dt)
    f["mu_phys_open"] = df["mu_pa_s"].values
    f["U_phys_open"] = df["U_w_m2k"].values
    f["A_phys_m2"] = df["A_wetted_m2"].values
    f["Q_jacket_phys_open_kw"] = df["Q_jacket_kw"].values
    f["hegman_phys"] = df["hegman_model"].values
    f["submergence_phys"] = df["submergence"].values
    f["hsd_kw_phys"] = df["hsd_kw"].values
    f["anchor_kw_phys"] = df["anchor_kw"].values
    f["energy_phys_cum_kwh"] = df["energy_cum_kwh"].values
    f["t_min"] = df["t_s"].values / 60.0
    f["step_idx"] = df["step"].map(STEP_IDX).values
    f["t_in_step_min"] = df.groupby("step").cumcount().values * dt / 60.0
    f["family_wb"] = int(recipe.family == "water_based")
    f["solids_pct"] = recipe.solids_pct
    f["pvc_pct"] = recipe.pvc_pct
    f["phi_millbase"] = (comp["m_pigment_kg"] / p.rho_p) / (comp["m_pigment_kg"] / p.rho_p + comp["m_charge_kg"] / p.rho_l)
    f["cp_final_kj_kgk"] = comp["cp_final_kj_kgk"]
    f["density_final_kg_l"] = comp["final_density_kg_l"]
    f["batch_mass_final_kg"] = recipe.batch_mass_kg
    f["mass_frac_of_final"] = df["batch_mass_kg"].values / recipe.batch_mass_kg
    f["hsd_rpm_setpoint"] = df["hsd_rpm_setpoint"].values
    f["anchor_rpm"] = df["anchor_rpm"].values
    f["T_jacket_in_c"] = df["T_jacket_in_c"].values
    f["cw_flow_m3h"] = df["cw_flow_m3h"].values
    f["T_ambient_c"] = df["T_ambient_c"].values
    return f[md.FEATURES_T_FORECAST]


def planned_schedule(t_target_min: float, planned_min: float, cfg: dict, stop_at_first_pass: bool) -> tuple[float, int, str]:
    """How long the line will actually disperse, given when the grind reaches target.
    Lab checks at first_check + k*interval, decision checks at the planned time and after each
    extension. Returns (dispersion_min, n_extensions, note)."""
    first, every = get(cfg, "data.lab_hegman_first_check_min"), get(cfg, "data.lab_hegman_interval_min")
    ext_min, max_ext = get(cfg, "simulation.operator.dispersion_extension_min"), get(cfg, "simulation.operator.max_extensions")
    if math.isnan(t_target_min):
        return planned_min + max_ext * ext_min, max_ext, "target not reached: runs all extensions, then QC hold"
    if stop_at_first_pass:
        checks = sorted({first + k * every for k in range(60)} | {planned_min + k * ext_min for k in range(max_ext + 1)})
        passing = [c for c in checks if c >= t_target_min]
        t = passing[0] if passing else planned_min + max_ext * ext_min
        n_ext = max(0, int(math.ceil((t - planned_min) / ext_min - 1e-9)))
        return t, n_ext, f"stops at the first passing lab check ({t:.0f} min)"
    if t_target_min <= planned_min + 1e-9:
        return planned_min, 0, "passes the lab check at the planned time"
    n_ext = min(int(math.ceil((t_target_min - planned_min) / ext_min - 1e-9)), max_ext)
    t = planned_min + n_ext * ext_min
    if planned_min + n_ext * ext_min < t_target_min - 1e-9:
        return t, n_ext, "target not reached even after the allowed extensions: QC hold"
    return t, n_ext, f"needs {n_ext} extension(s) of {ext_min:.0f} min"


def run_scenario(s: Settings, cfg: dict, twin: Twin | None = None) -> ScenarioResult:
    """Evaluate one set of settings. See the module docstring for the steps."""
    twin = twin or Twin()
    notes: list[str] = []
    recipe = _recipe(s)
    p = ph.PhysicsParams(cfg, recipe.family)
    comp = ph.composition(recipe, p)
    cond = _conditions(s)
    faults = ph.Faults(blade_wear=s.blade_wear, jacket_fouling_m2k_w=s.jacket_fouling_m2k_w)
    ext_min, max_ext = get(cfg, "simulation.operator.dispersion_extension_min"), get(cfg, "simulation.operator.max_extensions")

    # ---- run 1: grind curve with the longest possible dispersion
    df1, s1 = ph.simulate_batch(recipe, _phases(s, cfg, s.dispersion_min + max_ext * ext_min), cond, cfg, faults)
    t_target_phys = s1["time_to_target_hegman_min"]
    t_target = t_target_phys * twin.grind_time_factor if not math.isnan(t_target_phys) else NAN
    disp_min, n_ext, sched_note = planned_schedule(t_target, s.dispersion_min, cfg, s.stop_at_first_pass)
    notes.append(sched_note)

    # ---- run 2: the schedule the line will actually run
    df, sm = ph.simulate_batch(recipe, _phases(s, cfg, disp_min), cond, cfg, faults)
    hsd = df[df["step"] == "high_speed_dispersion"]
    t0 = float(hsd["t_s"].iloc[0]) / 60.0

    # ---- temperature: physics + learned residual
    T_phys = df["T_product_c"].values
    if twin.temperature is not None:
        T_pred = twin.temperature.predict_forecast(_forecast_features(df, recipe, s, cfg).assign(T_phys_open=T_phys))
    else:
        T_pred = T_phys.copy()
        notes.append("no residual model loaded: temperature is physics only")

    # ---- energy with the calibrated plant/physics ratio
    e_phys = float(df["energy_cum_kwh"].iloc[-1])
    e_pred = e_phys * twin.energy_factor
    e_hsd = float(((df["hsd_kw"].values[1:] + df["hsd_kw"].values[:-1]) * 0.5 * p.dt / 3600.0).sum()) * twin.energy_factor
    mass_t = recipe.batch_mass_kg / 1000.0

    # ---- grind: the calibrated curve is the physics curve with dispersion time scaled by the
    #      grind-time factor, H_cal(t0 + tau) = H_phys(t0 + tau / factor), read from run 1 (which
    #      dispersed for the maximum time, so the lookup never runs off the end)
    t1 = df1["t_s"].values / 60.0
    h1 = df1["hegman_model"].values
    t_min_arr = df["t_s"].values / 60.0
    in_hsd = (df["step"] == "high_speed_dispersion").values
    heg_curve = df["hegman_model"].values.copy()
    tau = np.where(in_hsd, (t_min_arr - t0) / twin.grind_time_factor, NAN)
    heg_curve[in_hsd] = np.interp(t0 + tau[in_hsd], t1, h1)
    after = np.where(~in_hsd & (t_min_arr > t0))[0]
    if after.size:
        heg_curve[after] = heg_curve[np.where(in_hsd)[0][-1]]          # no grind progress in letdown
    heg_end = float(np.interp(t0 + disp_min / twin.grind_time_factor, t1, h1))     # at the very end of the phase

    # ---- limits
    limit = p.T_limit
    margin = get(cfg, f"products.{recipe.family}.temp_alarm_margin_c")
    peak = float(np.max(T_pred))
    t_peak = float(df["t_s"].values[int(np.argmax(T_pred))]) / 60.0
    over = np.where(T_pred > limit)[0]
    t_breach = float(df["t_s"].values[over[0]]) / 60.0 if over.size else NAN
    v_lo, v_hi = get(cfg, "equipment.hsd.tip_speed_target_ms")
    v_tip = ph.tip_speed_ms(s.hsd_rpm, p.D)
    load_peak = 100.0 * float(df["hsd_kw"].max()) * twin.energy_factor / p.hsd_rated_kw
    limited = float(hsd["motor_limited"].mean()) * 100.0
    rpm_min = float(hsd["hsd_rpm"].min())
    subm = float(hsd["submergence"].min())
    v_lo_mass, v_hi_mass = get(cfg, "batch.mass_range_t")

    checks = pd.DataFrame([
        {"check": "product temperature below limit", "value": round(peak, 1), "limit": f"<= {limit}", "unit": "degC", "pass": peak <= limit,
         "severity": "FAIL"},
        {"check": "product temperature below pre-alarm", "value": round(peak, 1), "limit": f"<= {limit - margin}", "unit": "degC",
         "pass": peak <= limit - margin, "severity": "WARN"},
        {"check": "grind target reached in the run", "value": round(heg_end, 2), "limit": f">= {s.target_hegman}", "unit": "Hegman",
         "pass": heg_end >= s.target_hegman - 1e-6, "severity": "FAIL"},
        {"check": "no extensions needed", "value": n_ext, "limit": "= 0", "unit": "extensions", "pass": n_ext == 0, "severity": "WARN"},
        {"check": "HSD set-point achievable (drive not at its kW limit)", "value": round(load_peak, 0), "limit": "<= 100",
         "unit": "% of 55 kW", "pass": load_peak <= 100.0 and limited < 1.0, "severity": "FAIL"},
        {"check": "HSD motor load below continuous rating", "value": round(load_peak, 0), "limit": f"<= {get(cfg, 'equipment.hsd.max_continuous_load_pct')}",
         "unit": "% of 55 kW", "pass": load_peak <= get(cfg, "equipment.hsd.max_continuous_load_pct"), "severity": "WARN"},
        {"check": "batch fits the working volume", "value": round(comp["final_volume_m3"] * 1000), "limit": f"<= {get(cfg, 'equipment.vessel.working_volume_l')}",
         "unit": "L", "pass": not comp["overcharged"], "severity": "FAIL"},
        {"check": "batch mass within the equipment range", "value": round(mass_t, 2), "limit": f"{v_lo_mass}-{v_hi_mass}", "unit": "t",
         "pass": v_lo_mass <= mass_t <= v_hi_mass, "severity": "WARN"},
        {"check": "tip speed in the dispersion band", "value": round(v_tip, 1), "limit": f"{v_lo}-{v_hi}", "unit": "m/s",
         "pass": v_lo <= v_tip <= v_hi, "severity": "WARN"},
        {"check": "blade fully submerged in dispersion", "value": round(subm, 2), "limit": ">= 1", "unit": "-", "pass": subm >= 0.999,
         "severity": "WARN"},
    ])
    fails = checks[(~checks["pass"]) & (checks["severity"] == "FAIL")]["check"].tolist()
    warns = checks[(~checks["pass"]) & (checks["severity"] == "WARN")]["check"].tolist()

    curve = pd.DataFrame({"t_min": df["t_s"].values / 60.0, "step": df["step"].values,
                          "T_phys_c": T_phys, "T_pred_c": T_pred, "T_jacket_in_c": df["T_jacket_in_c"].values,
                          "T_jacket_out_c": df["T_jacket_out_c"].values, "hegman": heg_curve,
                          "hsd_rpm": df["hsd_rpm"].values, "hsd_kw": df["hsd_kw"].values * twin.energy_factor,
                          "anchor_kw": df["anchor_kw"].values,
                          "energy_cum_kwh": df["energy_cum_kwh"].values * twin.energy_factor,
                          "specific_energy_cum_kwh_t": df["energy_cum_kwh"].values * twin.energy_factor / mass_t,
                          "mu_pa_s": df["mu_pa_s"].values, "Q_jacket_kw": df["Q_jacket_kw"].values})
    summary = {
        "verdict": "PASS" if not fails else "FAIL", "fails": fails, "warnings": warns,
        "energy_total_kwh": round(e_pred, 2), "energy_hsd_kwh": round(e_hsd, 2), "specific_energy_kwh_per_t": round(e_pred / mass_t, 3),
        "peak_temp_c": round(peak, 2), "peak_at_min": round(t_peak, 1), "temp_limit_c": limit, "margin_to_limit_c": round(limit - peak, 2),
        "breach_at_min": None if math.isnan(t_breach) else round(t_breach, 1),
        "time_to_target_hegman_min": None if math.isnan(t_target) else round(t_target, 1),
        "time_to_target_physics_min": None if math.isnan(t_target_phys) else round(t_target_phys, 1),
        "dispersion_min": round(disp_min, 1), "planned_dispersion_min": s.dispersion_min, "n_extensions": n_ext,
        "hegman_end_of_dispersion": round(heg_end, 2), "target_hegman": s.target_hegman,
        "batch_cycle_min": round(sm["batch_cycle_min"], 1), "throughput_t_per_h": round(mass_t / (sm["batch_cycle_min"] / 60.0), 2),
        "hsd_load_peak_pct": round(load_peak, 0), "tip_speed_ms": round(v_tip, 1), "hsd_rpm_min_actual": round(rpm_min, 0),
        "final_volume_l": round(comp["final_volume_m3"] * 1000), "calibration": {"energy_factor": twin.energy_factor,
                                                                               "grind_time_factor": twin.grind_time_factor},
    }
    return ScenarioResult(s, summary, curve, checks, notes)


def compare_scenarios(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    """Side-by-side table of up to any number of named scenarios."""
    keys = ["verdict", "energy_total_kwh", "specific_energy_kwh_per_t", "peak_temp_c", "margin_to_limit_c", "breach_at_min",
            "time_to_target_hegman_min", "dispersion_min", "n_extensions", "hegman_end_of_dispersion", "batch_cycle_min",
            "throughput_t_per_h", "hsd_load_peak_pct", "tip_speed_ms"]
    rows = {}
    for name, r in results.items():
        col = {k: r.summary[k] for k in keys}
        col["fails"] = "; ".join(r.summary["fails"]) or "-"
        col["warnings"] = "; ".join(r.summary["warnings"]) or "-"
        for k in ["hsd_rpm", "anchor_rpm", "dispersion_min", "cw_inlet_c", "cw_flow_m3h", "batch_mass_kg", "solids_pct", "pvc_pct"]:
            col[f"set_{k}"] = getattr(r.settings, k)
        rows[name] = col
    return pd.DataFrame(rows)


# ============================================================ optimiser ==
@dataclass
class Bounds:
    hsd_rpm: tuple[float, float] = (764.0, 1061.0)       # the 18-25 m/s band
    rpm_step: float = 25.0
    anchor_rpm: tuple[float, float] = (30.0, 40.0)
    anchor_step: float = 5.0
    cw_flow_m3h: tuple[float, float] = (5.0, 15.0)
    flow_step: float = 2.5
    dispersion_min: tuple[float, float] = (20.0, 60.0)   # planned time is chosen to just reach the target
    safety_margin_c: float = 1.0                          # peak must stay this far under the limit
    grind_margin_frac: float = 0.15                       # planned time = predicted time to target x (1 + this): lot allowance
    max_load_pct: float = 95.0                            # planning rule: do not plan to run the drive at its limit
    allow_extensions: bool = False                        # a schedule that relies on extensions does not count


def optimise(base: Settings, cfg: dict, twin: Twin | None = None, bounds: Bounds | None = None,
             verbose: bool = False) -> dict:
    """Find the settings with the lowest kWh/t that pass every FAIL check, keep the peak at least
    safety_margin_c under the limit and reach the grind target without extensions.
    Strategy: coarse grid over rpm x anchor x flow; for each point the planned dispersion time is
    the calibrated time to target rounded up to the next half minute (plus the lab-check
    schedule); then a fine pass around the best point. Everything is explainable: the result
    carries the full table of candidates and why each was rejected."""
    twin = twin or Twin()
    b = bounds or Bounds()
    limit = get(cfg, f"products.{base.family}.temp_limit_c")
    rows = []

    def evaluate(rpm, anchor, flow, tag):
        # first pass: how long does the grind take at these settings?
        probe = run_scenario(replace(base, hsd_rpm=rpm, anchor_rpm=anchor, cw_flow_m3h=flow, dispersion_min=b.dispersion_min[1],
                                     stop_at_first_pass=False), cfg, twin)
        t_target = probe.summary["time_to_target_hegman_min"]
        if t_target is None or t_target > b.dispersion_min[1]:
            rows.append({"pass": tag, "hsd_rpm": rpm, "anchor_rpm": anchor, "cw_flow_m3h": flow, "dispersion_min": NAN,
                         "kwh_per_t": NAN, "peak_temp_c": probe.summary["peak_temp_c"], "feasible": False,
                         "reason": "grind target not reached within the maximum dispersion time"})
            return None
        plan = max(b.dispersion_min[0], math.ceil(t_target * (1.0 + b.grind_margin_frac) * 2) / 2)   # + lot allowance, next half minute
        r = run_scenario(replace(base, hsd_rpm=rpm, anchor_rpm=anchor, cw_flow_m3h=flow, dispersion_min=plan), cfg, twin)
        sm = r.summary
        reasons = list(sm["fails"])
        if sm["peak_temp_c"] > limit - b.safety_margin_c:
            reasons.append(f"peak {sm['peak_temp_c']:.1f} degC inside the {b.safety_margin_c} degC safety margin")
        if sm["hsd_load_peak_pct"] > b.max_load_pct:
            reasons.append(f"HSD load {sm['hsd_load_peak_pct']:.0f} % above the {b.max_load_pct:.0f} % planning limit")
        if not b.allow_extensions and sm["n_extensions"] > 0:
            reasons.append("needs extensions")
        rows.append({"pass": tag, "hsd_rpm": rpm, "anchor_rpm": anchor, "cw_flow_m3h": flow, "dispersion_min": plan,
                     "kwh_per_t": sm["specific_energy_kwh_per_t"], "kwh": sm["energy_total_kwh"], "peak_temp_c": sm["peak_temp_c"],
                     "batch_cycle_min": sm["batch_cycle_min"], "hsd_load_peak_pct": sm["hsd_load_peak_pct"],
                     "feasible": not reasons, "reason": "; ".join(reasons) or "ok"})
        return r

    rpms = np.arange(b.hsd_rpm[0], b.hsd_rpm[1] + 1e-9, b.rpm_step)
    anchors = np.arange(b.anchor_rpm[0], b.anchor_rpm[1] + 1e-9, b.anchor_step)
    flows = np.arange(b.cw_flow_m3h[0], b.cw_flow_m3h[1] + 1e-9, b.flow_step)
    for rpm in rpms:
        for anchor in anchors:
            for flow in flows:
                evaluate(float(rpm), float(anchor), float(flow), "coarse")
    table = pd.DataFrame(rows)
    feas = table[table["feasible"]]
    if feas.empty:
        return {"status": "no feasible settings in the bounds", "best": None, "candidates": table,
                "explanation": "Every candidate fails a hard check or sits inside the safety margin. Widen the bounds "
                               "(flow, anchor), accept extensions, or lower the batch size / raise the chilled-water capacity."}
    feas = feas.assign(kwh_r=feas["kwh_per_t"].round(2))
    best = feas.sort_values(["kwh_r", "peak_temp_c", "batch_cycle_min"]).iloc[0]
    # fine pass around the best: +/- one coarse step at half resolution
    for rpm in np.arange(best["hsd_rpm"] - b.rpm_step, best["hsd_rpm"] + b.rpm_step + 1e-9, b.rpm_step / 2):
        for flow in np.arange(best["cw_flow_m3h"] - b.flow_step, best["cw_flow_m3h"] + b.flow_step + 1e-9, b.flow_step / 2):
            if b.hsd_rpm[0] <= rpm <= b.hsd_rpm[1] and b.cw_flow_m3h[0] <= flow <= b.cw_flow_m3h[1]:
                evaluate(float(rpm), float(best["anchor_rpm"]), float(flow), "fine")
    table = pd.DataFrame(rows)
    feas = table[table["feasible"]].assign(kwh_r=lambda d: d["kwh_per_t"].round(2))
    best = feas.sort_values(["kwh_r", "peak_temp_c", "batch_cycle_min"]).iloc[0]
    best_settings = replace(base, hsd_rpm=float(best["hsd_rpm"]), anchor_rpm=float(best["anchor_rpm"]),
                            cw_flow_m3h=float(best["cw_flow_m3h"]), dispersion_min=float(best["dispersion_min"]))
    result = run_scenario(best_settings, cfg, twin)
    # what bound the answer? compare with the unconstrained-by-temperature best
    table_sorted = table.dropna(subset=["kwh_per_t"]).sort_values("kwh_per_t")
    cheapest = table_sorted.iloc[0]
    if cheapest["feasible"]:
        binding = "nothing binds: the lowest-energy point passes every check"
    else:
        binding = f"the lowest-energy point ({cheapest['hsd_rpm']:.0f} rpm, {cheapest['kwh_per_t']:.2f} kWh/t) is rejected: {cheapest['reason']}"
    return {"status": "ok", "best": result, "best_row": best.to_dict(), "candidates": table,
            "n_evaluated": int(table["dispersion_min"].notna().sum()), "n_feasible": int(table["feasible"].sum()),
            "binding": binding,
            "explanation": (f"Lowest kWh/t that passes every check: {best['kwh_per_t']:.2f} kWh/t at {best['hsd_rpm']:.0f} rpm, "
                            f"anchor {best['anchor_rpm']:.0f} rpm, chilled water {best['cw_flow_m3h']:.1f} m3/h, planned dispersion "
                            f"{best['dispersion_min']:.1f} min (predicted time to target + {100 * b.grind_margin_frac:.0f} % lot allowance); "
                            f"predicted peak {best['peak_temp_c']:.1f} degC (limit {limit}, safety margin {b.safety_margin_c}). {binding}. "
                            f"Caveat: the energy saving from higher tip speed rests on the assumed grind-effectiveness curve "
                            f"(physics.grind.tip_speed_ref_ms / effectiveness_exponent) - confirm it on real batches before "
                            f"changing recipes.")}


def save_calibration(path: Path | str, energy_factor: float, grind_time_factor: float, n_batches: int, note: str) -> None:
    Path(path).write_text(json.dumps({"energy_factor": energy_factor, "grind_time_factor": grind_time_factor,
                                      "n_batches": n_batches, "note": note}, indent=2))
