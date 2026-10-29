"""
simulate.py - generates a synthetic plant year of disperser batches (Step 3).

What it adds on top of the physics (Step 2):
  * a RECIPE CATALOGUE  - 12 products, water- and solvent-based, with realistic masses,
                          solids, PVC, grind targets and nominal settings
  * a CALENDAR + CLIMATE - batches spread over a year; hall temperature, chilled-water
                          inlet and raw-material temperature follow the season
  * OPERATOR VARIATION  - set-points and phase durations scatter around the recipe;
                          high-speed dispersion ends when the LAB CHECK passes (or after
                          the allowed extensions), the way a real line is run
  * RAW-MATERIAL LOTS   - solids, grindability, viscosity and achievable grind scatter
  * SENSOR NOISE        - every tag gets its own noise / offset / resolution
  * INJECTED FAULTS     - blade wear (slow sawtooth over blade life), jacket fouling
                          (slow build-up + dirty-water episodes), low cooling flow,
                          overcharged batch. The true fault state is written to a
                          separate truth file and the batch table, never to the
                          measured file.

Outputs (data/raw/):
  timeseries.csv        measured tags, 10 s     - what the plant will give the twin
  timeseries_truth.csv  hidden states, 10 s     - validation only
  batches.csv           one row per batch       - recipe, conditions, outcome, faults
  recipes.csv           the catalogue
plus data/dictionary/data_dictionary.csv.

All numbers come from the 'simulation' section of config/assumptions.yaml.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import kpis, physics as ph, schema
from .config import get

NAN = float("nan")


# ======================================================= plant vs model ==
def plant_config(cfg: dict) -> dict:
    """A copy of the config with the hidden plant-vs-model offsets applied. The
    SIMULATED PLANT runs on this; the twin runs on the unmodified config."""
    pc = copy.deepcopy(cfg)
    mm = cfg["simulation"]["plant_mismatch"]
    pc["physics"]["hsd_power"]["np_turbulent"]["value"] *= mm["np_turbulent_factor"]["value"]
    pc["physics"]["heat_transfer"]["h_product_ref_w_m2k"]["value"] *= mm["h_product_factor"]["value"]
    for fam in ("water_based", "solvent_based"):
        pc["physics"]["losses"]["evaporative_loss_w_per_k"][fam]["value"] *= mm["evaporative_loss_factor"]["value"]
        pc["physics"]["grind"]["char_specific_energy_kwh_t"][fam]["value"] *= mm["char_specific_energy_factor"]["value"]
    pc["physics"]["viscosity"]["phi_max"]["value"] += mm["phi_max_delta"]["value"]
    return pc


# ============================================================= catalogue ==
@dataclass
class CatalogueRecipe:
    recipe_id: str
    name: str
    family: str
    batch_mass_kg: float
    solids_pct: float
    pvc_pct: float
    target_hegman: float
    hegman_plateau: float      # best achievable grind for a normal lot
    dispersion_rpm: float      # recipe HSD speed (set-point)
    dispersion_min: float      # recipe HSD time (minimum before the lab check)
    pigment_min: float         # pigment addition time (scales with pigment mass)
    viscosity_ku: float        # nominal Stormer viscosity after letdown
    weight: float              # relative frequency within its family


CATALOGUE: list[CatalogueRecipe] = [
    # water-based ---------------------------------------------------------------------------
    CatalogueRecipe("WB-MATT-INT",   "Interior matt emulsion",      "water_based",   3000, 58, 55, 6.0, 6.9,  912, 38, 20, 105, 0.30),
    CatalogueRecipe("WB-SHEEN",      "Interior sheen emulsion",     "water_based",   2600, 50, 28, 6.5, 7.2,  950, 38, 15,  95, 0.16),
    CatalogueRecipe("WB-EXT-SMOOTH", "Exterior smooth emulsion",    "water_based",   3200, 56, 42, 6.0, 7.0,  912, 40, 18, 110, 0.16),
    CatalogueRecipe("WB-PRIMER",     "Water-based primer sealer",   "water_based",   3600, 60, 65, 5.0, 6.3,  920, 38, 22, 100, 0.11),
    CatalogueRecipe("WB-PUTTY",      "High-PVC wall putty",         "water_based",   4400, 68, 75, 4.5, 5.8,  900, 40, 28, 120, 0.07),
    CatalogueRecipe("WB-ECONOMY",    "Economy matt",                "water_based",   3800, 62, 70, 4.5, 5.9,  880, 38, 24, 115, 0.13),
    CatalogueRecipe("WB-DEEPBASE",   "Deep tint base",              "water_based",   2400, 46, 20, 7.0, 7.5, 1000, 42, 12,  90, 0.07),
    # solvent-based -------------------------------------------------------------------------
    CatalogueRecipe("SB-GLOSS-WHITE", "Alkyd gloss enamel white",   "solvent_based", 3000, 65, 18, 7.0, 7.7,  912, 40, 18,  85, 0.33),
    CatalogueRecipe("SB-GLOSS-COL",   "Alkyd gloss enamel colour",  "solvent_based", 2400, 60, 15, 7.0, 7.6,  950, 45, 15,  80, 0.17),
    CatalogueRecipe("SB-UNDERCOAT",   "Alkyd undercoat",            "solvent_based", 3200, 68, 32, 6.0, 7.0,  880, 40, 20,  90, 0.22),
    CatalogueRecipe("SB-REDOXIDE",    "Red oxide metal primer",     "solvent_based", 3400, 70, 35, 5.5, 6.6,  920, 38, 22,  92, 0.17),
    CatalogueRecipe("SB-SATIN",       "Alkyd satin enamel",         "solvent_based", 2000, 58, 22, 6.5, 7.4, 1000, 40, 12,  78, 0.11),
]


def catalogue_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in CATALOGUE])


# =============================================================== climate ==
class Climate:
    """Season -> hall temperature, chilled-water inlet, raw-material temperature."""

    def __init__(self, cfg: dict):
        c = "simulation.climate"
        self.amb_mean = get(cfg, f"{c}.ambient_mean_c")
        self.amb_amp = get(cfg, f"{c}.ambient_amplitude_c")
        self.hot_day = get(cfg, f"{c}.ambient_hottest_day_of_year")
        self.bump = get(cfg, f"{c}.afternoon_bump_c")
        self.amb_sd = get(cfg, f"{c}.ambient_noise_sd_c")
        self.cw_at20 = get(cfg, f"{c}.cw_inlet_at_20c_ambient")
        self.cw_slope = get(cfg, f"{c}.cw_inlet_slope_per_k_ambient")
        self.cw_sd = get(cfg, f"{c}.cw_inlet_noise_sd_c")
        self.cw_cycle_amp = get(cfg, f"{c}.cw_inlet_cycling_amplitude_c")
        self.cw_cycle_period_s = 60.0 * get(cfg, f"{c}.cw_inlet_cycling_period_min")
        self.raw_off = get(cfg, f"{c}.raw_material_offset_c")
        self.raw_sd = get(cfg, f"{c}.raw_material_noise_sd_c")
        self.cw_lo, self.cw_hi = get(cfg, "equipment.cooling_jacket.inlet_temp_range_c")

    def sample(self, start: datetime, rng: np.random.Generator) -> dict:
        doy = start.timetuple().tm_yday
        T_amb = (self.amb_mean + self.amb_amp * math.cos(2 * math.pi * (doy - self.hot_day) / 365.0)
                 + (self.bump if 12 <= start.hour < 18 else 0.0) + rng.normal(0, self.amb_sd))
        T_cw = float(np.clip(self.cw_at20 + self.cw_slope * (T_amb - 20.0) + rng.normal(0, self.cw_sd),
                             self.cw_lo, self.cw_hi))
        T_raw = T_amb + self.raw_off + rng.normal(0, self.raw_sd)
        phase = rng.uniform(0, 2 * math.pi)
        amp, period = self.cw_cycle_amp, self.cw_cycle_period_s
        return {"T_ambient_c": T_amb, "T_cw_in_c": T_cw, "T_raw_c": T_raw,
                "T_cw_in_fn": (lambda t: T_cw + amp * math.sin(2 * math.pi * t / period + phase))}


# ================================================================ faults ==
class FaultSchedule:
    """Fault state for every batch index, decided up front so the whole year is consistent:
    blade wear is a sawtooth over the blade life, fouling a sawtooth between cleanings plus
    random dirty-water episodes, low flow and overcharge are per-batch events."""

    def __init__(self, cfg: dict, n: int, rng: np.random.Generator):
        f = "simulation.faults"
        life = get(cfg, f"{f}.blade_life_batches")
        wear_end = get(cfg, f"{f}.blade_wear_at_end_of_life")
        clean = get(cfg, f"{f}.fouling_cleaning_interval_batches")
        foul_end = get(cfg, f"{f}.fouling_at_cleaning_m2k_w")
        p_ep = get(cfg, f"{f}.fouling_episode_probability")
        ep_extra = get(cfg, f"{f}.fouling_episode_extra_m2k_w")
        ep_lo, ep_hi = get(cfg, f"{f}.fouling_episode_length_batches")
        p_lf = get(cfg, f"{f}.low_flow_probability")
        lf_lo, lf_hi = get(cfg, f"{f}.low_flow_fraction_range")
        p_oc = get(cfg, f"{f}.overcharge_probability")
        oc_lo, oc_hi = get(cfg, f"{f}.overcharge_fraction_range")
        dosing_sd = get(cfg, "simulation.lot_variation.mass_dosing_sd_frac")
        self.wear_thr = get(cfg, f"{f}.blade_wear_label_threshold")
        self.foul_thr = get(cfg, f"{f}.fouling_label_threshold_m2k_w")

        self.wear = np.array([wear_end * (i % life) / life for i in range(n)])
        base_foul = np.array([foul_end * (i % clean) / clean for i in range(n)])
        episode = np.zeros(n)
        i = 0
        while i < n:
            if rng.random() < p_ep:
                length = int(rng.integers(ep_lo, ep_hi + 1))
                for j in range(i, min(n, i + length)):
                    if j % clean == 0 and j > i:      # a cleaning ends the episode
                        break
                    episode[j] = ep_extra
                i += length
            else:
                i += 1
        self.fouling = base_foul + episode
        self.flow_frac = np.where(rng.random(n) < p_lf, rng.uniform(lf_lo, lf_hi, n), 1.0)
        self.mass_frac = np.where(rng.random(n) < p_oc, rng.uniform(oc_lo, oc_hi, n),
                                  rng.normal(1.0, dosing_sd, n))

    def labels(self, i: int) -> tuple[str, str]:
        active = []
        if self.wear[i] >= self.wear_thr:
            active.append("BLADE_WEAR")
        if self.fouling[i] >= self.foul_thr:
            active.append("JACKET_FOULING")
        if self.flow_frac[i] < 1.0:
            active.append("LOW_CW_FLOW")
        if self.mass_frac[i] > 1.06:
            active.append("OVERCHARGE")
        if not active:
            return "NONE", ""
        return (active[0] if len(active) == 1 else "MULTIPLE"), "+".join(active)


# ======================================================== one batch run ==
def run_one_batch(i: int, rec: CatalogueRecipe, start: datetime, cfg: dict, climate: Climate,
                  faults: FaultSchedule, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Simulate one batch end to end. Returns (measured timeseries, truth timeseries, batch row)."""
    op, lot, sen = "simulation.operator", "simulation.lot_variation", "simulation.sensors"
    g = lambda path: get(cfg, path)                                                       # noqa: E731
    dt = float(g("data.sample_period_s"))

    # ---- lot and dosing scatter
    mass = rec.batch_mass_kg * faults.mass_frac[i]
    solids = rec.solids_pct + rng.normal(0, g(f"{lot}.solids_sd_pct"))
    recipe = ph.Recipe(rec.family, mass, solids, rec.pvc_pct, rec.target_hegman, rec.recipe_id,
                       hegman_plateau=rec.hegman_plateau + rng.normal(0, g(f"{lot}.hegman_plateau_sd")),
                       grind_energy_factor=float(np.exp(rng.normal(0, g(f"{lot}.grind_energy_sd_frac")))),
                       viscosity_factor=float(np.exp(rng.normal(0, g(f"{lot}.viscosity_sd_frac")))))

    # ---- climate and utilities
    cl = climate.sample(start, rng)
    flow_choices = g(f"{op}.cw_flow_choices_m3h")
    flow_sp = float(rng.choice(flow_choices, p=g(f"{op}.cw_flow_choice_probabilities")))
    flow_actual = flow_sp * faults.flow_frac[i]
    cond = ph.Conditions(T0_c=cl["T_raw_c"], T_raw_c=cl["T_raw_c"], T_cw_in_c=cl["T_cw_in_c"],
                         cw_flow_m3h=flow_actual, T_ambient_c=cl["T_ambient_c"], T_cw_in_fn=cl["T_cw_in_fn"])
    flt = ph.Faults(blade_wear=float(faults.wear[i]), jacket_fouling_m2k_w=float(faults.fouling[i]))

    # ---- operator set-points
    def lognorm(mean, sd_frac):
        return float(mean * np.exp(rng.normal(0, sd_frac)))

    rpm_lo, rpm_hi = g("process_steps.high_speed_dispersion.hsd_rpm_range")
    hsd_sp = float(np.clip(rec.dispersion_rpm * (1 + rng.normal(0, g(f"{op}.rpm_setpoint_sd_frac"))), rpm_lo, rpm_hi))
    anchor_off = rng.normal(0, g(f"{op}.anchor_rpm_sd"))

    def anchor_for(step):
        lo, hi = g(f"process_steps.{step}.anchor_rpm_range")
        return float(np.clip(0.5 * (lo + hi) + anchor_off, lo, hi))

    def mid(path):
        lo, hi = g(path)
        return 0.5 * (lo + hi)

    dur_sd = g(f"{op}.duration_sd_frac")
    planned_disp_min = lognorm(rec.dispersion_min, g(f"{op}.planned_dispersion_sd_frac"))
    ext_min, max_ext = g(f"{op}.dispersion_extension_min"), g(f"{op}.max_extensions")
    first_check, check_int = g("data.lab_hegman_first_check_min"), g("data.lab_hegman_interval_min")
    hegman_sd = g(f"{sen}.hegman_lab_noise_sd")

    p = ph.PhysicsParams(cfg, rec.family)
    comp = ph.composition(recipe, p)
    phases = [
        ph.Phase("charge_liquids", 60 * lognorm(mid("process_steps.charge_liquids.duration_range_min"), dur_sd),
                 mid("process_steps.charge_liquids.hsd_rpm_range"), anchor_for("charge_liquids"), comp["m_charge_kg"], "liquid"),
        ph.Phase("pigment_addition", 60 * lognorm(rec.pigment_min, dur_sd),
                 mid("process_steps.pigment_addition.hsd_rpm_range"), anchor_for("pigment_addition"), comp["m_pigment_kg"], "pigment"),
        ph.Phase("high_speed_dispersion", 60 * (planned_disp_min + max_ext * ext_min),
                 hsd_sp, anchor_for("high_speed_dispersion"), 0.0, "none"),
        ph.Phase("letdown", 60 * lognorm(mid("process_steps.letdown.duration_range_min"), dur_sd),
                 mid("process_steps.letdown.hsd_rpm_range"), anchor_for("letdown"), comp["m_letdown_kg"], "letdown"),
    ]

    # ---- run, with the lab-check stop rule on the high-speed phase
    sim = ph.BatchSimulator(recipe, cond, cfg, flt)
    sim.run_phase(phases[0])
    sim.run_phase(phases[1])

    t_hsd0 = sim.t
    check_times = sorted({first_check + k * check_int for k in range(50) if first_check + k * check_int < planned_disp_min}
                         | {planned_disp_min + k * ext_min for k in range(max_ext + 1)})
    lab: dict[float, float] = {}          # absolute t_s of the row -> reading
    state = {"next": 0, "n_ext": 0, "last_reading": NAN}

    def take_reading(H_true: float) -> float:
        return float(np.clip(round((H_true + rng.normal(0, hegman_sd)) * 2) / 2, 0.0, 8.0))

    def stop_rule(row: dict) -> bool:
        elapsed_min = (row["t_s"] - t_hsd0 + dt) / 60.0
        stop = False
        while state["next"] < len(check_times) and elapsed_min >= check_times[state["next"]] - 1e-9:
            t_check = check_times[state["next"]]
            reading = take_reading(row["hegman_model"])
            lab[row["t_s"]] = reading
            state["last_reading"] = reading
            state["next"] += 1
            if t_check >= planned_disp_min - 1e-9:                # a decision check
                if reading >= rec.target_hegman:
                    stop = True
                elif state["n_ext"] < max_ext:
                    state["n_ext"] += 1
                else:
                    stop = True                                    # give up: QC hold
        return stop

    sim.run_phase(phases[2], stop_rule=stop_rule)
    disp_min = (sim.t - t_hsd0) / 60.0
    sim.run_phase(phases[3])
    truth_df, summary = sim.finish()

    # ---- measured layer: sensor noise, offsets, resolution, sparse lab columns
    n = len(truth_df)
    meas = pd.DataFrame({
        "batch_id": f"TSD-{i + 1:04d}",
        "timestamp": [start + timedelta(seconds=float(t)) for t in truth_df["t_s"]],
        "t_s": truth_df["t_s"].values,
        "step": truth_df["step"].values,
        "product_family": rec.family, "recipe_id": rec.recipe_id,
        "batch_mass_kg": np.round((truth_df["batch_mass_kg"] + rng.normal(0, g(f"{sen}.mass_noise_sd_kg"), n))
                                  / g(f"{sen}.mass_resolution_kg")) * g(f"{sen}.mass_resolution_kg"),
        "solids_pct": round(solids, 1), "pvc_pct": rec.pvc_pct, "target_hegman": rec.target_hegman,
        "hsd_rpm_setpoint": truth_df["hsd_rpm_setpoint"].values,
        "hsd_rpm": np.round(truth_df["hsd_rpm"] + rng.normal(0, g(f"{sen}.rpm_noise_sd"), n)),
        "anchor_rpm": np.round(truth_df["anchor_rpm"].values, 0),
        "hsd_kw": np.round(truth_df["hsd_kw"] * (1 + rng.normal(0, g(f"{sen}.kw_noise_frac"), n)) / g(f"{sen}.kw_resolution_kw"))
                  * g(f"{sen}.kw_resolution_kw"),
        "anchor_kw": np.round(truth_df["anchor_kw"] * (1 + rng.normal(0, g(f"{sen}.kw_noise_frac"), n)) / g(f"{sen}.kw_resolution_kw"))
                     * g(f"{sen}.kw_resolution_kw"),
        "T_product_c": truth_df["T_product_c"] + rng.normal(0, g(f"{sen}.T_product_offset_sd_c")) + rng.normal(0, g(f"{sen}.T_product_noise_sd_c"), n),
        "T_jacket_in_c": truth_df["T_jacket_in_c"] + rng.normal(0, g(f"{sen}.T_jacket_noise_sd_c"), n),
        "T_jacket_out_c": truth_df["T_jacket_out_c"] + rng.normal(0, g(f"{sen}.T_jacket_noise_sd_c"), n),
        "cw_flow_m3h": truth_df["cw_flow_m3h"] * (1 + rng.normal(0, g(f"{sen}.flow_noise_frac"), n)),
        "T_ambient_c": cl["T_ambient_c"] + rng.normal(0, 0.3, n),
        "hegman": [lab.get(t, NAN) for t in truth_df["t_s"]],
        "viscosity_ku": NAN,
    })
    for c in ["T_product_c", "T_jacket_in_c", "T_jacket_out_c", "cw_flow_m3h", "T_ambient_c"]:
        meas[c] = meas[c].round(2)
    ku = rec.viscosity_ku + 1.2 * (solids - rec.solids_pct) + rng.normal(0, g(f"{sen}.viscosity_lab_noise_sd_ku"))
    meas.loc[meas.index[-1], "viscosity_ku"] = round(ku, 1)

    truth = pd.DataFrame({
        "batch_id": meas["batch_id"], "t_s": truth_df["t_s"].values,
        "T_true_c": truth_df["T_product_c"].values, "hegman_true": truth_df["hegman_model"].values,
        "hsd_kw_true": truth_df["hsd_kw"].values,
    })
    for c in ["mu_pa_s", "wetted_frac", "phi", "reynolds", "power_number", "motor_limited", "U_w_m2k",
              "A_wetted_m2", "Q_jacket_kw", "Q_loss_kw", "volume_m3", "density_kg_l", "cp_kj_kgk", "energy_cum_kwh"]:
        truth[c] = truth_df[c].values

    # ---- batch row (measured outcome + hidden truth)
    e_total = kpis.energy_kwh(meas["hsd_kw"], meas["t_s"]) + kpis.energy_kwh(meas["anchor_kw"], meas["t_s"])
    t_lab = kpis.dispersion_time_min(meas, rec.target_hegman)
    lab_readings = meas["hegman"].dropna()
    peak_meas = float(meas["T_product_c"].max())
    label, active = faults.labels(i)
    row = {
        "batch_id": meas["batch_id"].iloc[0], "start_time": start.isoformat(timespec="seconds"), "month": start.month,
        "recipe_id": rec.recipe_id, "recipe_name": rec.name, "product_family": rec.family,
        "batch_mass_nominal_kg": rec.batch_mass_kg, "batch_mass_kg": float(meas["batch_mass_kg"].iloc[-1]),
        "solids_pct": round(solids, 1), "pvc_pct": rec.pvc_pct, "target_hegman": rec.target_hegman,
        "hsd_rpm_setpoint": round(hsd_sp, 0), "anchor_rpm": phases[2].anchor_rpm, "cw_flow_setpoint_m3h": flow_sp,
        "T_cw_in_mean_c": round(float(meas["T_jacket_in_c"].mean()), 2), "T_ambient_c": round(cl["T_ambient_c"], 1),
        "T_raw_c": round(cl["T_raw_c"], 1),
        "planned_dispersion_min": round(planned_disp_min, 1), "dispersion_min": round(disp_min, 1),
        "n_extensions": state["n_ext"], "batch_cycle_min": round(summary["batch_cycle_min"], 1),
        "time_to_target_hegman_lab_min": t_lab,
        "hegman_lab_final": float(lab_readings.iloc[-1]) if not lab_readings.empty else NAN,
        "hegman_reached_lab": bool(lab_readings.ge(rec.target_hegman).any()),
        "viscosity_ku": round(ku, 1),
        "energy_total_kwh": round(e_total, 2), "specific_energy_kwh_per_t": round(e_total / (mass / 1000.0), 3),
        "peak_temp_c": round(peak_meas, 2), "breached": peak_meas > p.T_limit,
        "hsd_load_peak_pct": round(100.0 * float(meas["hsd_kw"].max()) / p.hsd_rated_kw, 1),
        # hidden truth
        "hegman_plateau": round(recipe.hegman_plateau, 2), "grind_energy_factor": round(recipe.grind_energy_factor, 3),
        "viscosity_factor": round(recipe.viscosity_factor, 3),
        "blade_wear": round(float(faults.wear[i]), 3), "jacket_fouling_m2k_w": round(float(faults.fouling[i]), 5),
        "cw_flow_actual_frac": round(float(faults.flow_frac[i]), 3), "overcharge_frac": round(float(faults.mass_frac[i]), 3),
        "fault_injected": label, "faults_active": active,
        "time_to_target_hegman_true_min": summary["time_to_target_hegman_min"],
        "peak_temp_true_c": round(summary["peak_temp_c"], 2), "breached_true": summary["breached"],
        "overcharged_volume": summary["overcharged"], "motor_limited_share_pct": round(summary["motor_limited_share_pct"], 1),
    }
    return meas, truth, row


# ============================================================= the year ==
def generate_dataset(cfg: dict, out_dir: Path | str, n_batches: int | None = None,
                     seed: int | None = None, verbose: bool = True) -> dict:
    """Generate the whole synthetic year and write the four CSVs. Returns a small log dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(n_batches or get(cfg, "simulation.batches_generated"))
    seed = int(seed if seed is not None else get(cfg, "simulation.random_seed"))
    rng = np.random.default_rng(seed)

    climate = Climate(cfg)
    faults = FaultSchedule(cfg, n, rng)
    plant_cfg = plant_config(cfg)              # the simulated plant is NOT the twin's physics
    share_wb = get(cfg, "simulation.family_share_water_based")
    wb = [r for r in CATALOGUE if r.family == "water_based"]
    sb = [r for r in CATALOGUE if r.family == "solvent_based"]
    w_wb = np.array([r.weight for r in wb]) / sum(r.weight for r in wb)
    w_sb = np.array([r.weight for r in sb]) / sum(r.weight for r in sb)

    start = datetime.fromisoformat(get(cfg, "simulation.calendar.start_date"))
    gap_days = get(cfg, "simulation.calendar.mean_days_between_batches")
    h_lo, h_hi = get(cfg, "simulation.calendar.start_hour_range")

    meas_parts, truth_parts, rows = [], [], []
    day = 0.0
    for i in range(n):
        day += rng.exponential(gap_days)
        t0 = (start + timedelta(days=math.floor(day))).replace(hour=int(rng.integers(h_lo, h_hi + 1)),
                                                               minute=int(rng.integers(0, 60)), second=0)
        rec = (wb[rng.choice(len(wb), p=w_wb)] if rng.random() < share_wb else sb[rng.choice(len(sb), p=w_sb)])
        meas, truth, row = run_one_batch(i, rec, t0, plant_cfg, climate, faults, rng)
        meas_parts.append(meas)
        truth_parts.append(truth)
        rows.append(row)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} batches simulated")

    timeseries = pd.concat(meas_parts, ignore_index=True)[schema.measured_columns()]
    truth = pd.concat(truth_parts, ignore_index=True)[schema.truth_columns()]
    batches = pd.DataFrame(rows)
    timeseries.to_csv(out_dir / "timeseries.csv", index=False, float_format="%.3f")
    truth.to_csv(out_dir / "timeseries_truth.csv", index=False, float_format="%.4f")
    batches.to_csv(out_dir / "batches.csv", index=False)
    catalogue_frame().to_csv(out_dir / "recipes.csv", index=False)
    schema.write_dictionary(out_dir.parent / "dictionary" / "data_dictionary.csv")
    log = {"seed": seed, "n_batches": n, "rows": int(len(timeseries)), "config_version": get(cfg, "meta.version"),
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "note": "SIMULATED data: Step 2 physics with hidden plant-vs-model offsets, noise and faults. Not plant data."}
    (out_dir / "generation_log.json").write_text(json.dumps(log, indent=2))
    return log
