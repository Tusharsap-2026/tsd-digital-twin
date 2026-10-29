"""
schema.py - the twin's own tag list: canonical column names for one batch.

Why this matters for the twin:
  The simulator (Step 3) writes these columns. The feature builder, models,
  what-if simulator and dashboard all READ these columns. In Step 9 the only
  thing that changes is a mapping table "PLC tag -> canonical column"; nothing
  downstream has to be rewritten. That is what makes the sim-to-live switch a
  configuration change rather than a rebuild.

Three groups:
  BATCH_TIMESERIES : what the plant measures (or the simulator fakes)      -> data/raw/timeseries.csv
  TRUTH_TIMESERIES : the simulator's hidden truth, for validation only     -> data/raw/timeseries_truth.csv
  BATCH_TABLE      : one row per batch (recipe, conditions, faults, outcome) -> data/raw/batches.csv
  TWIN_OUTPUTS     : what the twin adds alongside
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

STEP_NAMES = ["charge_liquids", "pigment_addition", "high_speed_dispersion", "letdown"]
PRODUCT_FAMILIES = ["water_based", "solvent_based"]
ALERT_CODES = ["NONE", "TEMP_PREALARM", "TEMP_LIMIT", "BLADE_WEAR", "JACKET_FOULING",
               "LOW_CW_FLOW", "OVERCHARGE", "MOTOR_OVERLOAD"]
FAULT_LABELS = ["NONE", "BLADE_WEAR", "JACKET_FOULING", "LOW_CW_FLOW", "OVERCHARGE", "MULTIPLE"]

# (column, unit, source, sampling, description)
BATCH_TIMESERIES = [
    ("batch_id",         "-",        "MES",              "per batch", "Unique batch identifier (MES batch number)"),
    ("timestamp",        "ISO-8601", "historian",        "10 s",      "Absolute time stamp, local plant time"),
    ("t_s",              "s",        "derived",          "10 s",      "Seconds since batch start (first row of charge_liquids)"),
    ("step",             "-",        "MES / PLC phase",  "10 s",      " | ".join(STEP_NAMES)),
    ("product_family",   "-",        "MES recipe",       "per batch", " | ".join(PRODUCT_FAMILIES)),
    ("recipe_id",        "-",        "MES recipe",       "per batch", "Recipe / product code"),
    ("batch_mass_kg",    "kg",       "PLC load cells",   "10 s",      "Mass in the vessel (rises during charging)"),
    ("solids_pct",       "% w/w",    "MES recipe",       "per batch", "Recipe solids content"),
    ("pvc_pct",          "%",        "MES recipe",       "per batch", "Pigment volume concentration"),
    ("target_hegman",    "Hegman",   "MES recipe",       "per batch", "Grind target for the millbase"),
    ("hsd_rpm_setpoint", "rpm",      "PLC / VFD",        "10 s",      "High-speed disperser speed set-point"),
    ("hsd_rpm",          "rpm",      "PLC / VFD",        "10 s",      "High-speed disperser actual shaft speed"),
    ("anchor_rpm",       "rpm",      "PLC / VFD",        "10 s",      "Anchor / scraper shaft speed"),
    ("hsd_kw",           "kW",       "PLC / VFD",        "10 s",      "HSD motor electrical power"),
    ("anchor_kw",        "kW",       "PLC / VFD",        "10 s",      "Anchor motor electrical power"),
    ("T_product_c",      "degC",     "PLC (vessel RTD)", "10 s",      "Product temperature"),
    ("T_jacket_in_c",    "degC",     "PLC",              "10 s",      "Chilled-water inlet temperature"),
    ("T_jacket_out_c",   "degC",     "PLC",              "10 s",      "Chilled-water outlet temperature"),
    ("cw_flow_m3h",      "m3/h",     "PLC",              "10 s",      "Chilled-water flow to the jacket"),
    ("T_ambient_c",      "degC",     "PLC or weather",   "10 s",      "Hall air temperature (assumed available; daily value if not)"),
    ("hegman",           "Hegman",   "LAB",              "sparse",    "Grind-gauge reading; NaN when not sampled"),
    ("viscosity_ku",     "KU",       "LAB",              "sparse",    "Stormer viscosity; NaN when not sampled"),
]

TRUTH_TIMESERIES = [
    ("batch_id",        "-",      "simulator", "per batch", "Join key"),
    ("t_s",             "s",      "simulator", "10 s",      "Join key"),
    ("T_true_c",        "degC",   "simulator", "10 s",      "Product temperature without sensor noise or offset"),
    ("hegman_true",     "Hegman", "simulator", "10 s",      "Continuous model grind (what the lab samples with noise)"),
    ("hsd_kw_true",     "kW",     "simulator", "10 s",      "HSD electrical power without sensor noise"),
    ("mu_pa_s",         "Pa.s",   "simulator", "10 s",      "Millbase viscosity at blade shear rate"),
    ("wetted_frac",     "-",      "simulator", "10 s",      "Fraction of pigment wetted"),
    ("phi",             "-",      "simulator", "10 s",      "Pigment volume fraction in the vessel"),
    ("reynolds",        "-",      "simulator", "10 s",      "Impeller Reynolds number"),
    ("power_number",    "-",      "simulator", "10 s",      "HSD power number"),
    ("motor_limited",   "bool",   "simulator", "10 s",      "Drive at its kW limit (speed sagging)"),
    ("U_w_m2k",         "W/m2K",  "simulator", "10 s",      "Overall heat-transfer coefficient incl. fouling"),
    ("A_wetted_m2",     "m2",     "simulator", "10 s",      "Jacket area in contact with product"),
    ("Q_jacket_kw",     "kW",     "simulator", "10 s",      "Heat removed by the jacket"),
    ("Q_loss_kw",       "kW",     "simulator", "10 s",      "Heat lost to the hall and evaporation"),
    ("volume_m3",       "m3",     "simulator", "10 s",      "Liquid volume in the vessel"),
    ("density_kg_l",    "kg/L",   "simulator", "10 s",      "Bulk density"),
    ("cp_kj_kgk",       "kJ/kgK", "simulator", "10 s",      "Bulk specific heat"),
    ("energy_cum_kwh",  "kWh",    "simulator", "10 s",      "Cumulative electrical energy, exact"),
]

BATCH_TABLE = [
    ("batch_id",                    "-",      "MES",       "Unique batch identifier"),
    ("start_time",                  "ISO-8601", "MES",     "Batch start"),
    ("month",                       "-",      "derived",   "Calendar month of the start (seasonality)"),
    ("recipe_id",                   "-",      "MES",       "Recipe code"),
    ("recipe_name",                 "-",      "MES",       "Recipe description"),
    ("product_family",              "-",      "MES",       "water_based | solvent_based"),
    ("batch_mass_nominal_kg",       "kg",     "MES",       "Recipe batch size"),
    ("batch_mass_kg",               "kg",     "PLC",       "Actual final mass (load cells)"),
    ("solids_pct",                  "% w/w",  "MES",       "Batch solids after lot scatter"),
    ("pvc_pct",                     "%",      "MES",       "Pigment volume concentration"),
    ("target_hegman",               "Hegman", "MES",       "Grind target"),
    ("hsd_rpm_setpoint",            "rpm",    "PLC",       "Operator set-point for high-speed dispersion"),
    ("anchor_rpm",                  "rpm",    "PLC",       "Anchor speed in high-speed dispersion"),
    ("cw_flow_setpoint_m3h",        "m3/h",   "PLC",       "Chilled-water flow set-point"),
    ("T_cw_in_mean_c",              "degC",   "PLC",       "Mean chilled-water inlet during the batch"),
    ("T_ambient_c",                 "degC",   "PLC",       "Hall temperature"),
    ("T_raw_c",                     "degC",   "derived",   "Raw-material temperature"),
    ("planned_dispersion_min",      "min",    "MES",       "Recipe HSD time after operator adjustment"),
    ("dispersion_min",              "min",    "MES",       "Actual HSD time incl. extensions"),
    ("n_extensions",                "-",      "MES",       "Extensions after failed lab checks"),
    ("batch_cycle_min",             "min",    "MES",       "Charge start to letdown end"),
    ("time_to_target_hegman_lab_min", "min",  "LAB",       "HSD start to first lab reading >= target (NaN = never)"),
    ("hegman_lab_final",            "Hegman", "LAB",       "Last lab grind reading"),
    ("hegman_reached_lab",          "bool",   "LAB",       "Target met on a lab reading"),
    ("viscosity_ku",                "KU",     "LAB",       "Stormer after letdown"),
    ("energy_total_kwh",            "kWh",    "derived",   "HSD + anchor electrical energy (from measured kW)"),
    ("specific_energy_kwh_per_t",   "kWh/t",  "derived",   "Energy per tonne of batch"),
    ("peak_temp_c",                 "degC",   "derived",   "Peak measured product temperature"),
    ("breached",                    "bool",   "derived",   "Measured temperature exceeded the family limit"),
    ("hsd_load_peak_pct",           "%",      "derived",   "Peak HSD load"),
    # ---- ground truth, simulator only (never available live) ----
    ("hegman_plateau",              "Hegman", "simulator", "Best achievable grind for this lot"),
    ("grind_energy_factor",         "-",      "simulator", "Pigment-lot grindability multiplier"),
    ("viscosity_factor",            "-",      "simulator", "Raw-material-lot viscosity multiplier"),
    ("blade_wear",                  "-",      "simulator", "Blade wear state 0..1"),
    ("jacket_fouling_m2k_w",        "m2K/W",  "simulator", "Jacket fouling resistance"),
    ("cw_flow_actual_frac",         "-",      "simulator", "Actual / set-point flow (<1 = low-flow fault)"),
    ("overcharge_frac",             "-",      "simulator", "Actual / nominal mass (>1.06 = overcharge fault)"),
    ("fault_injected",              "-",      "simulator", " | ".join(FAULT_LABELS)),
    ("time_to_target_hegman_true_min", "min", "simulator", "HSD start to true (continuous) Hegman crossing the target"),
    ("peak_temp_true_c",            "degC",   "simulator", "Peak true product temperature"),
    ("breached_true",               "bool",   "simulator", "True temperature exceeded the limit"),
    ("overcharged_volume",          "bool",   "simulator", "Final volume exceeded the working volume"),
    ("motor_limited_share_pct",     "%",      "simulator", "Share of HSD time at the drive limit"),
]

TWIN_OUTPUTS = [
    ("T_pred_c",                  "degC",   "twin",           "10 s",      "Predicted product temperature"),
    ("hsd_kw_pred",               "kW",     "twin",           "10 s",      "Predicted HSD power"),
    ("energy_cum_kwh",            "kWh",    "derived",        "10 s",      "Cumulative energy, HSD + anchor"),
    ("specific_energy_cum_kwh_t", "kWh/t",  "derived",        "10 s",      "Cumulative specific energy"),
    ("hegman_pred",               "Hegman", "twin",           "10 s",      "Predicted grind development"),
    ("breach_eta_min",            "min",    "twin",           "10 s",      "Predicted minutes until the temperature limit (NaN = none)"),
    ("alert_code",                "-",      "twin",           "10 s",      " | ".join(ALERT_CODES)),
]

COLUMNS = ["column", "unit", "source", "sampling", "description"]


def as_dataframe() -> pd.DataFrame:
    rows = [dict(zip(COLUMNS, r), table="timeseries") for r in BATCH_TIMESERIES]
    rows += [dict(zip(COLUMNS, r), table="timeseries_truth") for r in TRUTH_TIMESERIES]
    rows += [dict(zip(["column", "unit", "source", "description"], r), sampling="per batch", table="batches")
             for r in BATCH_TABLE]
    rows += [dict(zip(COLUMNS, r), table="twin_output") for r in TWIN_OUTPUTS]
    return pd.DataFrame(rows)[["table"] + COLUMNS]


def write_dictionary(path: Path | str) -> Path:
    """Write the data dictionary CSV and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    as_dataframe().to_csv(path, index=False)
    return path


def measured_columns() -> list[str]:
    return [r[0] for r in BATCH_TIMESERIES]


def truth_columns() -> list[str]:
    return [r[0] for r in TRUTH_TIMESERIES]


def batch_columns() -> list[str]:
    return [r[0] for r in BATCH_TABLE]


def twin_columns() -> list[str]:
    return [r[0] for r in TWIN_OUTPUTS]
