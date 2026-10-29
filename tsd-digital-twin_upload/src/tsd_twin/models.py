"""
models.py - the twin's prediction models (Step 5).

Three questions, three model families, every one of them EXPLAINABLE:

1. TEMPERATURE over the batch (row level, 10 s). Five predictors, from "knows nothing
   measured" to "knows everything measured so far":
     physics          T_phys_open: Step 2 replayed with the set-points and utilities
     ml_only          gradient boosting on measured tags, no physics inputs (the foil)
     hybrid_forecast  T_phys_open + GB(residual | physics states, recipe, set-points, utilities)
                      -> no measured product-side tags: usable for WHAT-IF (Step 6)
     hybrid_nowcast   + measured kW, jacket outlet, cooling duty  -> the live display
     online           + what was measured HORIZON minutes earlier (lag_*) -> best nowcast with history
     ahead            hybrid_forecast inputs + lag_* only: nothing measured at the target time,
                      so it is a TRUE 5-min-ahead forecast -> ALERTS (Step 7/8)

2. ENERGY per batch (kWh, kWh/t) and TIME TO TARGET HEGMAN (batch level), known at the
   start of high-speed dispersion + 5 min of evidence:
     energy_given_schedule  ratio actual / physics for the schedule ACTUALLY run
                            (the what-if question: "this recipe, these settings, this time")
     energy_forecast        ratio actual / physics for the PLANNED schedule
                            (the pre-batch question: it also has to guess the lab outcome)
     time_to_target         minutes from HSD start to the first lab reading >= target
     peak_temp              offset actual - physics peak; breach = peak > limit
   Each as a ridge regression (the simple baseline) and as HistGradientBoosting with
   monotone constraints where the physics insists on a direction.

The GB models predict a CORRECTION to the physics (ratio or offset), never the quantity
itself, so they stay anchored to the physics outside the training range. Feature
importance is permutation importance on TEST batches: what the model really uses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import kpis
from .config import get

NAN = float("nan")

# ================================================================ features ==
# Temperature, row level ------------------------------------------------------
ROW_PHYSICS_STATE = ["T_phys_open", "dT_phys_to_jacket_c", "dTdt_phys_c_per_min", "mu_phys", "U_phys",
                     "A_phys_m2", "Q_jacket_phys_kw", "hegman_phys", "submergence_phys", "hsd_kw_phys",
                     "anchor_kw_phys", "energy_phys_cum_kwh"]
# the same states from the OPEN run only (no measured power): what a hypothetical what-if batch can supply
ROW_PHYSICS_STATE_OPEN = ["T_phys_open", "dT_phys_open_to_jacket_c", "dTdt_phys_open_c_per_min", "mu_phys_open",
                          "U_phys_open", "A_phys_m2", "Q_jacket_phys_open_kw", "hegman_phys", "submergence_phys",
                          "hsd_kw_phys", "anchor_kw_phys", "energy_phys_cum_kwh"]
ROW_RECIPE_SETPOINTS = ["t_min", "step_idx", "t_in_step_min", "family_wb", "solids_pct", "pvc_pct", "phi_millbase",
                        "cp_final_kj_kgk", "density_final_kg_l", "batch_mass_final_kg", "mass_frac_of_final",
                        "hsd_rpm_setpoint", "anchor_rpm", "T_jacket_in_c", "cw_flow_m3h", "T_ambient_c"]
ROW_MEASURED_PRODUCT = ["hsd_rpm", "tip_speed_ms", "hsd_kw", "anchor_kw", "hsd_kw_resid_phys", "anchor_kw_resid_phys",
                        "T_jacket_out_c", "cooling_duty_meas_kw", "energy_cum_kwh", "specific_energy_cum_kwh_t",
                        "hsd_kw_per_t_millbase"]
ROW_LAG = ["lag_T_meas_c", "lag_resid_open_c", "lag_resid_replay_c", "lag_dTdt_meas_c_per_min", "lag_T_phys_replay_c"]
FEATURES_T_ML_ONLY = ROW_RECIPE_SETPOINTS + ROW_MEASURED_PRODUCT                 # no physics anywhere
FEATURES_T_FORECAST = ROW_PHYSICS_STATE_OPEN + ROW_RECIPE_SETPOINTS              # what-if capable
FEATURES_T_NOWCAST = ROW_PHYSICS_STATE + ROW_RECIPE_SETPOINTS + ROW_MEASURED_PRODUCT
FEATURES_T_ONLINE = FEATURES_T_NOWCAST + ROW_LAG
FEATURES_T_AHEAD = FEATURES_T_FORECAST + ROW_LAG       # true 5-min-ahead: nothing measured at the target time
T_MODEL_NAMES = ["physics", "ml_only", "hybrid_forecast", "hybrid_nowcast", "online", "ahead"]

# Batch level ------------------------------------------------------------------
BATCH_PRE = ["family_wb", "batch_mass_kg", "solids_pct", "pvc_pct", "target_hegman", "millbase_mass_kg",
             "phi_millbase", "cp_final_kj_kgk", "density_final_kg_l", "hsd_rpm_setpoint", "tip_speed_setpoint_ms",
             "anchor_rpm_hsd", "cw_flow_m3h_at_t0", "T_cw_in_at_t0_c", "T_ambient_c", "T_at_hsd_start_c",
             "planned_dispersion_min", "charge_min", "pigment_min", "energy_at_hsd_start_kwh"]
BATCH_EVIDENCE = ["pigment_kw_resid_phys_pct", "pigment_anchor_kw_resid_pct", "pigment_T_resid_replay_c",
                  "hsd5_kw_mean", "hsd5_kw_resid_phys_pct", "hsd5_dTdt_meas_c_per_min", "hsd5_dTdt_phys_c_per_min",
                  "hsd5_T_resid_replay_c", "hsd5_rpm_sag", "hsd5_cooling_duty_kw"]
BATCH_PHYSICS = ["phys_energy_total_kwh", "phys_specific_energy_kwh_per_t", "phys_time_to_target_min",
                 "phys_hegman_end", "phys_peak_temp_c", "phys_hsd_load_peak_pct", "phys_hsd_kw_mean_dispersion",
                 "phys_anchor_kw_mean_dispersion"]
FEATURES_BATCH = BATCH_PRE + BATCH_EVIDENCE + BATCH_PHYSICS
FEATURES_BATCH_SCHED = FEATURES_BATCH + ["y_dispersion_min", "phys_energy_actual_sched_kwh"]   # schedule given

# monotone constraints the physics insists on (+1 = output rises with feature, -1 falls, 0 free)
MONO_ENERGY = {"hsd5_kw_resid_phys_pct": 1, "pigment_anchor_kw_resid_pct": 1}
MONO_TIME = {"hsd_rpm_setpoint": -1, "target_hegman": 1, "hsd5_kw_resid_phys_pct": -1, "phys_time_to_target_min": 1}
MONO_PEAK = {"T_cw_in_at_t0_c": 1, "cw_flow_m3h_at_t0": -1, "hsd_rpm_setpoint": 1, "planned_dispersion_min": 1,
             "phys_peak_temp_c": 1, "T_at_hsd_start_c": 1, "hsd5_T_resid_replay_c": 1}


def _gb(seed: int, monotone: dict | None = None, features: list[str] | None = None, **kw) -> HistGradientBoostingRegressor:
    params = dict(max_iter=400, learning_rate=0.05, max_depth=4, min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=False, random_state=seed)
    params.update(kw)
    if monotone and features:
        params["monotonic_cst"] = [monotone.get(f, 0) for f in features]
    return HistGradientBoostingRegressor(**params)


# ============================================================ temperature ==
@dataclass
class TemperatureModels:
    """physics, ML-only, hybrid forecast, hybrid nowcast and online temperature predictors."""
    seed: int = 42
    ml_only: HistGradientBoostingRegressor | None = None
    forecast: HistGradientBoostingRegressor | None = None
    nowcast: HistGradientBoostingRegressor | None = None
    online: HistGradientBoostingRegressor | None = None
    ahead: HistGradientBoostingRegressor | None = None

    def fit(self, rows: pd.DataFrame) -> "TemperatureModels":
        r = rows
        self.ml_only = _gb(self.seed, max_iter=300).fit(r[FEATURES_T_ML_ONLY], r["T_product_c"])
        self.forecast = _gb(self.seed, max_iter=300).fit(r[FEATURES_T_FORECAST], r["resid_open_c"])
        self.nowcast = _gb(self.seed, max_iter=300).fit(r[FEATURES_T_NOWCAST], r["resid_open_c"])
        ok = r[ROW_LAG].notna().all(axis=1)
        self.online = _gb(self.seed, max_iter=300).fit(r.loc[ok, FEATURES_T_ONLINE], r.loc[ok, "resid_open_c"])
        self.ahead = _gb(self.seed, max_iter=300).fit(r.loc[ok, FEATURES_T_AHEAD], r.loc[ok, "resid_open_c"])
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Five temperature predictions per row, in degC."""
        out = pd.DataFrame(index=rows.index)
        base = rows["T_phys_open"].values
        out["T_physics"] = base
        out["T_ml_only"] = self.ml_only.predict(rows[FEATURES_T_ML_ONLY])
        out["T_hybrid_forecast"] = base + self.forecast.predict(rows[FEATURES_T_FORECAST])
        out["T_hybrid_nowcast"] = base + self.nowcast.predict(rows[FEATURES_T_NOWCAST])
        ok = rows[ROW_LAG].notna().all(axis=1)
        online = np.full(len(rows), NAN)
        if ok.any():
            online[ok.values] = base[ok.values] + self.online.predict(rows.loc[ok, FEATURES_T_ONLINE])
        out["T_online"] = online
        out["T_online"] = out["T_online"].fillna(out["T_hybrid_nowcast"])   # no history yet -> nowcast
        ahead = np.full(len(rows), NAN)
        if ok.any():
            ahead[ok.values] = base[ok.values] + self.ahead.predict(rows.loc[ok, FEATURES_T_AHEAD])
        out["T_ahead"] = ahead
        out["T_ahead"] = out["T_ahead"].fillna(out["T_hybrid_forecast"])    # no history yet -> forecast
        return out

    def predict_ahead(self, rows: pd.DataFrame) -> np.ndarray:
        """5-min-ahead entry point: rows carry the physics states of the TARGET time and lag_* = now."""
        return rows["T_phys_open"].values + self.ahead.predict(rows[FEATURES_T_AHEAD])

    def predict_forecast(self, rows: pd.DataFrame) -> np.ndarray:
        """What-if entry point: physics states + recipe + set-points only."""
        return rows["T_phys_open"].values + self.forecast.predict(rows[FEATURES_T_FORECAST])


def temperature_scorecard(rows: pd.DataFrame, pred: pd.DataFrame, cfg: dict, label_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-batch errors for every predictor, and the fleet roll-up with pass rates."""
    tgt_mae = get(cfg, "kpi_targets.temp_mae_c")
    tgt_max = get(cfg, "kpi_targets.temp_max_abs_error_c")
    recs = []
    for bid, idx in rows.groupby("batch_id").groups.items():
        T = rows.loc[idx, "T_product_c"].values
        rec = {"batch_id": bid}
        if label_col:
            rec[label_col] = rows.loc[idx, label_col].iloc[0]
        for name in T_MODEL_NAMES:
            p = pred.loc[idx, f"T_{name}"].values
            e = kpis.temperature_errors(T, p)
            rec[f"{name}_mae"] = e["mae_c"]
            rec[f"{name}_max"] = e["max_abs_error_c"]
            rec[f"{name}_bias"] = e["bias_c"]
            rec[f"{name}_peak_err"] = float(np.nanmax(p) - T.max())     # the number that decides breaches
        recs.append(rec)
    per_batch = pd.DataFrame(recs)
    fleet_rows = []
    for name in T_MODEL_NAMES:
        mae, mx, pk = per_batch[f"{name}_mae"], per_batch[f"{name}_max"], per_batch[f"{name}_peak_err"]
        fleet_rows.append({"model": name, "n_batches": len(per_batch),
                           "mae_median": mae.median(), "mae_mean": mae.mean(), "mae_p90": mae.quantile(0.9), "mae_worst": mae.max(),
                           f"pass_mae<={tgt_mae}": 100 * (mae <= tgt_mae).mean(),
                           "maxerr_p90": mx.quantile(0.9), f"pass_max<={tgt_max}": 100 * (mx <= tgt_max).mean(),
                           "peak_bias": pk.mean(), "peak_mae": pk.abs().mean()})
    return per_batch, pd.DataFrame(fleet_rows)


# ================================================================ batch ==
@dataclass
class BatchModels:
    """Energy (schedule given / forecast), time-to-target-Hegman and peak-temperature predictors."""
    seed: int = 42
    features: list[str] = field(default_factory=lambda: list(FEATURES_BATCH))
    features_sched: list[str] = field(default_factory=lambda: list(FEATURES_BATCH_SCHED))
    energy_sched_lin: object = None
    energy_sched_gb: HistGradientBoostingRegressor | None = None
    energy_fc_lin: object = None
    energy_fc_gb: HistGradientBoostingRegressor | None = None
    time_lin: object = None
    time_gb: HistGradientBoostingRegressor | None = None
    peak_lin: object = None
    peak_gb: HistGradientBoostingRegressor | None = None
    medians: pd.Series | None = None

    def _X(self, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        return df[feats].copy().fillna(self.medians[feats])

    @staticmethod
    def _lin():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))

    def fit(self, b: pd.DataFrame) -> "BatchModels":
        self.medians = b[self.features_sched].median()
        X, Xs = self._X(b, self.features), self._X(b, self.features_sched)
        gb_kw = dict(max_iter=300, max_depth=3, min_samples_leaf=10)
        # energy given the schedule: ratio to physics for the schedule actually run
        y = b["y_energy_total_kwh"] / b["phys_energy_actual_sched_kwh"]
        self.energy_sched_lin = self._lin().fit(Xs, y)
        self.energy_sched_gb = _gb(self.seed, MONO_ENERGY, self.features_sched, **gb_kw).fit(Xs, y)
        # energy forecast at t0: ratio to physics for the PLANNED schedule (lab outcome unknown)
        y = b["y_energy_total_kwh"] / b["phys_energy_total_kwh"]
        self.energy_fc_lin = self._lin().fit(X, y)
        self.energy_fc_gb = _gb(self.seed, MONO_ENERGY, self.features, **gb_kw).fit(X, y)
        # time to target (minutes), on batches whose lab reached the target
        ok = b["y_time_to_target_lab_min"].notna()
        y = b.loc[ok, "y_time_to_target_lab_min"]
        self.time_lin = self._lin().fit(X[ok], y)
        self.time_gb = _gb(self.seed, MONO_TIME, self.features, **gb_kw).fit(X[ok], y)
        # peak temperature: offset actual - physics
        y = b["y_peak_temp_c"] - b["phys_peak_temp_c"]
        self.peak_lin = self._lin().fit(X, y)
        self.peak_gb = _gb(self.seed, MONO_PEAK, self.features, **gb_kw).fit(X, y)
        return self

    def predict(self, b: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        X, Xs = self._X(b, self.features), self._X(b, self.features_sched)
        out = pd.DataFrame({"batch_id": b["batch_id"].values})
        mass_t = b["batch_mass_kg"].values / 1000.0
        # energy given schedule
        ps = b["phys_energy_actual_sched_kwh"].values
        out["esched_physics"] = ps
        out["esched_linear"] = ps * self.energy_sched_lin.predict(Xs)
        out["esched_gb"] = ps * self.energy_sched_gb.predict(Xs)
        # energy forecast
        pf = b["phys_energy_total_kwh"].values
        out["efc_physics"] = pf
        out["efc_linear"] = pf * self.energy_fc_lin.predict(X)
        out["efc_gb"] = pf * self.energy_fc_gb.predict(X)
        for k in ["physics", "linear", "gb"]:
            out[f"kwh_t_sched_{k}"] = out[f"esched_{k}"] / mass_t
            out[f"kwh_t_fc_{k}"] = out[f"efc_{k}"] / mass_t
        # time to target
        out["time_physics"] = b["phys_time_to_target_min"].values
        out["time_linear"] = self.time_lin.predict(X)
        out["time_gb"] = self.time_gb.predict(X)
        # two-stage forecast: which lab check will pass -> schedule -> energy for that schedule
        ext_min = float(get(cfg, "simulation.operator.dispersion_extension_min"))
        max_ext = int(get(cfg, "simulation.operator.max_extensions"))
        planned = b["planned_dispersion_min"].values
        for k in ["linear", "gb"]:
            t_pred = out[f"time_{k}"].values
            n_ext = np.clip(np.ceil((t_pred - planned) / ext_min), 0, max_ext).astype(int)
            disp = planned + n_ext * ext_min
            phys_sched = pf + (b["phys_hsd_kw_mean_dispersion"].values + b["phys_anchor_kw_mean_dispersion"].values) * n_ext * ext_min / 60.0
            Xh = Xs.copy()
            Xh["y_dispersion_min"] = disp
            Xh["phys_energy_actual_sched_kwh"] = phys_sched
            model = self.energy_sched_gb if k == "gb" else self.energy_sched_lin
            out[f"n_ext_pred_{k}"] = n_ext
            out[f"disp_pred_{k}"] = disp
            out[f"efc2_{k}"] = phys_sched * model.predict(Xh)
            out[f"kwh_t_fc2_{k}"] = out[f"efc2_{k}"] / mass_t
        # peak and breach
        pk = b["phys_peak_temp_c"].values
        out["peak_physics"] = pk
        out["peak_linear"] = pk + self.peak_lin.predict(X)
        out["peak_gb"] = pk + self.peak_gb.predict(X)
        wb = b["family_wb"].values == 1
        limit = np.where(wb, get(cfg, "products.water_based.temp_limit_c"), get(cfg, "products.solvent_based.temp_limit_c"))
        margin = np.where(wb, get(cfg, "products.water_based.temp_alarm_margin_c"), get(cfg, "products.solvent_based.temp_alarm_margin_c"))
        out["limit_c"] = limit
        for k in ["physics", "linear", "gb"]:
            out[f"breach_{k}"] = out[f"peak_{k}"] > limit
            out[f"prealarm_{k}"] = out[f"peak_{k}"] > limit - margin
        return out


def batch_scorecard(b: pd.DataFrame, pred: pd.DataFrame, cfg: dict, truth_time: pd.Series | None = None) -> pd.DataFrame:
    """Energy / time / peak accuracy for the three predictors with the KPI targets.
    truth_time (simulator only): continuous time-to-target, for a second, non-quantised score."""
    tgt_e, tgt_t = get(cfg, "kpi_targets.energy_error_pct"), get(cfg, "kpi_targets.dispersion_time_error_pct")
    m = b.merge(pred, on="batch_id")
    if truth_time is not None:
        m["time_true"] = m["batch_id"].map(truth_time)
    rows = []
    for k in ["physics", "linear", "gb"]:
        es = 100 * (m[f"esched_{k}"] - m["y_energy_total_kwh"]) / m["y_energy_total_kwh"]
        ef = 100 * (m[f"efc_{k}"] - m["y_energy_total_kwh"]) / m["y_energy_total_kwh"]
        no_ext = m["y_n_extensions"] == 0
        if f"efc2_{k}" in m.columns:
            ef2 = 100 * (m[f"efc2_{k}"] - m["y_energy_total_kwh"]) / m["y_energy_total_kwh"]
            ext_right = (m[f"n_ext_pred_{k}"] == m["y_n_extensions"])
        else:
            ef2, ext_right = ef * np.nan, pd.Series(np.nan, index=m.index)
        ok = m["y_time_to_target_lab_min"].notna() & m[f"time_{k}"].notna()
        d = m.loc[ok, f"time_{k}"] - m.loc[ok, "y_time_to_target_lab_min"]
        t_pct = 100 * d / m.loc[ok, "y_time_to_target_lab_min"]
        pk = m[f"peak_{k}"] - m["y_peak_temp_c"]
        act, prd = m["y_breached"].astype(bool), m[f"breach_{k}"].astype(bool)
        row = {"model": k, "n": len(m),
               "esched_mape": es.abs().mean(), "esched_bias": es.mean(), "esched_p90": es.abs().quantile(0.9),
               f"esched_pass<={tgt_e}%": 100 * (es.abs() <= tgt_e).mean(),
               "efc_mape": ef.abs().mean(), "efc_bias": ef.mean(), f"efc_pass<={tgt_e}%": 100 * (ef.abs() <= tgt_e).mean(),
               "efc_mape_no_ext": ef[no_ext].abs().mean(), "efc_mape_ext": ef[~no_ext].abs().mean(),
               "efc2_mape": ef2.abs().mean(), "efc2_bias": ef2.mean(), f"efc2_pass<={tgt_e}%": 100 * (ef2.abs() <= tgt_e).mean(),
               "efc2_schedule_right_%": 100 * ext_right.mean(),
               "efc2_mape_sched_right": ef2[ext_right.fillna(False).astype(bool)].abs().mean(),
               "time_n": int(ok.sum()), "time_mae_min": d.abs().mean(), "time_bias_min": d.mean(),
               "time_mape": t_pct.abs().mean(), f"time_pass<={tgt_t}%": 100 * (t_pct.abs() <= tgt_t).mean(),
               "time_within_one_check": 100 * (d.abs() <= 10.0).mean(),
               "time_unpredicted": int((m["y_time_to_target_lab_min"].notna() & m[f"time_{k}"].isna()).sum()),
               "peak_mae": pk.abs().mean(), "peak_bias": pk.mean(),
               "breach_actual": int(act.sum()), "breach_hits": int((act & prd).sum()),
               "breach_misses": int((act & ~prd).sum()), "breach_false": int((~act & prd).sum()),
               "prealarm_catches": int((act & m[f"prealarm_{k}"].astype(bool)).sum()),
               "prealarm_false": int((~act & m[f"prealarm_{k}"].astype(bool)).sum())}
        if truth_time is not None:
            ok2 = m["time_true"].notna() & m[f"time_{k}"].notna()
            d2 = m.loc[ok2, f"time_{k}"] - m.loc[ok2, "time_true"]
            row["time_true_mae_min"] = d2.abs().mean()
            row[f"time_true_pass<={tgt_t}%"] = 100 * ((100 * d2 / m.loc[ok2, "time_true"]).abs() <= tgt_t).mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ importance ==
def importance(model, X: pd.DataFrame, y: pd.Series, seed: int = 42, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance on held-out data: how much the MAE grows when a feature is shuffled."""
    ok = y.notna()
    r = permutation_importance(model, X[ok], y[ok], n_repeats=n_repeats, random_state=seed, scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "sd": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# ================================================================ persist ==
def save_models(tm: TemperatureModels, bm: BatchModels, out_dir: Path | str, meta: dict) -> None:
    import joblib
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tm, out_dir / "temperature_models.joblib")
    joblib.dump(bm, out_dir / "batch_models.joblib")
    (out_dir / "model_card.json").write_text(json.dumps(meta, indent=2, default=str))


def load_models(out_dir: Path | str) -> tuple[TemperatureModels, BatchModels, dict]:
    import joblib
    out_dir = Path(out_dir)
    return (joblib.load(out_dir / "temperature_models.joblib"), joblib.load(out_dir / "batch_models.joblib"),
            json.loads((out_dir / "model_card.json").read_text()))
