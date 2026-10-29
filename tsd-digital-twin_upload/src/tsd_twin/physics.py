"""
physics.py - first-principles model of the twin-shaft disperser (Step 2, restructured in Step 3).

What is modelled, in plain words
--------------------------------
1. POWER DRAW of the high-speed disperser (HSD) and the anchor.
     P_shaft = Np * rho * N^3 * D^5          [W]   N in rev/s, D in m, rho in kg/m3
     Np      = Kp / Re + Np_turb             power number: high in thick (laminar) product,
     Re      = rho * N * D^2 / mu            flat in thin (turbulent) product
   The VFD cannot deliver more than the motor's rated kW, so the ACTUAL speed
   sags below the set-point in thick millbase (torque / current limit).

2. VISCOSITY of the millbase at blade shear rate.
     mu            = mu_vehicle(T) * (1 - phi_eff/phi_max)^(-n)      Krieger-Dougherty
     mu_vehicle(T) = mu_ref * exp(-b (T - T_ref))                    thinner when hot
     phi_eff       = phi * (1 + k_agg * (1 - w))                      unwetted agglomerates trap liquid
     dw/dt         = (1 - w) * e_dot / e_wet                          w = wetted fraction, e_dot = shaft kWh/t per s

3. ENERGY BALANCE of the product mass, stepped forward at the sample period.
     m cp dT/dt = eta * (P_hsd + P_anchor) - Q_jacket - Q_loss   (+ enthalpy of material added)
     Q_jacket   = U A (T - T_cw,in) / (1 + U A / (2 m_dot cp_w))   jacket water warms as it passes
     1/U        = 1/h_product + wall/k + 1/h_jacket + R_fouling
     Q_loss     = U_loss A_loss (T - T_amb) + k_evap * max(T - T_amb, 0)

4. GRIND development (Hegman) driven by effective specific energy.
     dH/dt = (H_plateau - H) * w * f(v) * (1 - wear) * e_dot / e_c
     f(v)  = max(0, (v - v_min) / (v_ref - v_min))^alpha               v = blade tip speed

Structure (Step 3): BatchSimulator holds the state and runs one phase at a time, so a
phase can be stopped by a rule (e.g. "stop high-speed dispersion when the lab Hegman
check passes"). simulate_batch() is the one-call wrapper used by the Step 2 checks.

Every number comes from config/assumptions.yaml. Nothing numeric is hard-coded here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .config import get

STEP_ORDER = ["charge_liquids", "pigment_addition", "high_speed_dispersion", "letdown"]
NAN = float("nan")


# ================================================================ inputs ==
@dataclass
class Recipe:
    family: str                     # "water_based" | "solvent_based"
    batch_mass_kg: float
    solids_pct: float               # total non-volatile content, % w/w of the finished batch
    pvc_pct: float                  # pigment volume concentration of the dry film
    target_hegman: float
    recipe_id: str = "REF"
    hegman_plateau: float | None = None     # None -> family default from the config
    grind_energy_factor: float = 1.0        # pigment lot: >1 = harder to grind (multiplies e_c)
    viscosity_factor: float = 1.0           # raw-material lot: multiplies the vehicle viscosity


@dataclass
class Phase:
    name: str                       # one of STEP_ORDER
    duration_s: float               # maximum duration; a stop rule may end it earlier
    hsd_rpm: float                  # set-point; actual may be lower (motor limit)
    anchor_rpm: float
    mass_added_kg: float = 0.0      # added linearly over duration_s
    added_kind: str = "none"        # "liquid" | "pigment" | "letdown" | "none"


@dataclass
class Conditions:
    T0_c: float = 28.0              # temperature of the first liquids charged
    T_raw_c: float = 28.0           # temperature of everything added after that
    T_cw_in_c: float = 18.0         # chilled-water inlet (constant, unless T_cw_in_fn is given)
    cw_flow_m3h: float = 10.0       # chilled-water flow (constant, unless cw_flow_fn is given)
    T_ambient_c: float = 30.0
    T_cw_in_fn: Callable[[float], float] | None = None   # optional: inlet temperature as a function of t_s
    cw_flow_fn: Callable[[float], float] | None = None   # optional: flow as a function of t_s
    hsd_kw_fn: Callable[[float], float] | None = None    # optional: MEASURED HSD electrical kW as a function of t_s
    #   (replay mode: the energy balance is driven by measured power instead of the power model)
    hsd_rpm_fn: Callable[[float], float] | None = None   # optional: measured actual rpm, used with hsd_kw_fn


@dataclass
class Faults:
    blade_wear: float = 0.0              # 0 = new blade; 0.3 = 30 % loss of grind effectiveness
    jacket_fouling_m2k_w: float = 0.0    # extra thermal resistance on the jacket side
    # "low cooling flow" is simply Conditions.cw_flow_m3h; "overcharge" is simply a
    # Recipe.batch_mass_kg above the recipe's nominal mass.


# ============================================================ parameters ==
class PhysicsParams:
    """Pull every physics number out of the config once, so the time loop stays readable."""

    def __init__(self, cfg: dict, family: str):
        g = lambda path: get(cfg, path)                        # noqa: E731
        self.family = family
        self.dt = float(g("data.sample_period_s"))
        # vessel
        self.vessel_D = g("equipment.vessel.inner_diameter_m")
        self.jacket_h = g("equipment.vessel.jacket_height_m")
        self.A_bottom = g("equipment.vessel.bottom_dish_area_m2")
        self.wall_m = g("equipment.vessel.wall_thickness_mm") / 1000.0
        self.working_vol_m3 = g("equipment.vessel.working_volume_l") / 1000.0
        # HSD
        self.D = g("equipment.hsd.blade_diameter_m")
        self.hsd_max_rpm = g("equipment.hsd.speed_max_rpm")
        self.hsd_rated_kw = g("equipment.hsd.motor_rated_kw")
        self.hsd_eff = g("equipment.hsd.motor_efficiency") * g("equipment.hsd.drive_efficiency")
        self.hsd_limit_kw = self.hsd_rated_kw * g("equipment.hsd.max_continuous_load_pct") / 100.0
        self.Np_turb = g("physics.hsd_power.np_turbulent")
        self.Kp = g("physics.hsd_power.laminar_constant_kp")
        self.hsd_no_load_kw = g("physics.hsd_power.no_load_kw")
        self.blade_h_nominal = g("physics.hsd_power.blade_height_above_bottom_m")
        self.blade_h_min = g("physics.hsd_power.blade_height_min_m")
        self.blade_h_frac = g("physics.hsd_power.blade_height_fraction_of_millbase")
        self.subm_ramp = g("physics.hsd_power.submergence_ramp_m")
        self.wear_power_factor = g("physics.hsd_power.blade_wear_power_factor")
        # anchor
        self.Da = g("physics.anchor_power.sweep_diameter_m")
        self.anchor_rated_kw = g("equipment.anchor.motor_rated_kw")
        self.anchor_eff = g("equipment.anchor.motor_efficiency") * g("equipment.anchor.drive_efficiency")
        self.Np_turb_a = g("physics.anchor_power.np_turbulent")
        self.Kp_a = g("physics.anchor_power.laminar_constant_kp")
        self.anchor_no_load_kw = g("physics.anchor_power.no_load_kw")
        self.min_eff_anchor_rpm = g("physics.anchor_power.min_effective_rpm")
        # cooling water
        self.rho_w = g("equipment.cooling_jacket.water_density_kg_m3")
        self.cp_w = g("equipment.cooling_jacket.water_cp_kj_kgk")
        # materials
        m = f"physics.materials.{family}"
        self.rho_p = g(f"{m}.pigment_density_kg_l")
        self.rho_b = g(f"{m}.binder_density_kg_l")
        self.rho_l = g(f"{m}.liquid_density_kg_l")
        self.cp_l = g(f"{m}.cp_liquid_kj_kgk")
        self.cp_b = g(f"{m}.cp_binder_solid_kj_kgk")
        self.cp_p = g(f"{m}.cp_pigment_kj_kgk")
        self.charge_frac = g(f"{m}.charge_fraction")
        # viscosity
        self.mu_ref = g(f"physics.viscosity.{family}.vehicle_mu_ref_pa_s")
        self.b = g(f"physics.viscosity.{family}.temp_coeff_per_k")
        self.T_ref = g("physics.viscosity.reference_temp_c")
        self.phi_max = g("physics.viscosity.phi_max")
        self.kd_n = g("physics.viscosity.kd_exponent")
        self.k_agg = g("physics.viscosity.agglomerate_factor")
        self.phi_cap = g("physics.viscosity.phi_eff_cap")
        self.e_wet = g("physics.viscosity.wetting_energy_kwh_t")
        # heat transfer
        h = "physics.heat_transfer"
        self.eta = g(f"{h}.dissipation_fraction")
        self.h_j_ref = g(f"{h}.h_jacket_ref_w_m2k")
        self.flow_ref = g(f"{h}.jacket_ref_flow_m3h")
        self.flow_exp = g(f"{h}.jacket_flow_exponent")
        self.k_wall = g(f"{h}.wall_conductivity_w_mk")
        self.h_p_ref = g(f"{h}.h_product_ref_w_m2k")
        self.anchor_ref_rpm = g(f"{h}.product_ref_anchor_rpm")
        self.mu_ref_ht = g(f"{h}.product_ref_mu_pa_s")
        self.anchor_exp = g(f"{h}.product_anchor_exponent")
        self.mu_exp = g(f"{h}.product_mu_exponent")
        # losses
        self.U_loss = g("physics.losses.u_loss_w_m2k")
        self.A_loss = g("physics.losses.a_loss_m2")
        self.k_evap = g(f"physics.losses.evaporative_loss_w_per_k.{family}")
        # grind
        self.H0 = g("physics.grind.hegman_initial")
        self.H_max = g(f"physics.grind.hegman_plateau.{family}")
        self.e_c = g(f"physics.grind.char_specific_energy_kwh_t.{family}")
        self.v_min = g("physics.grind.tip_speed_min_ms")
        self.v_ref = g("physics.grind.tip_speed_ref_ms")
        self.alpha = g("physics.grind.effectiveness_exponent")
        # product limit
        self.T_limit = g(f"products.{family}.temp_limit_c")


# ========================================================== composition ==
def composition(recipe: Recipe, p: PhysicsParams) -> dict:
    """Split the batch into pigment, binder solids and liquid, and decide what goes in when.
    Pigment share of the solids follows from PVC and the two densities:
        pigment mass share = PVC*rho_p / (PVC*rho_p + (1-PVC)*rho_b)"""
    s = recipe.solids_pct / 100.0
    pvc = recipe.pvc_pct / 100.0
    m = recipe.batch_mass_kg
    pig_share = pvc * p.rho_p / (pvc * p.rho_p + (1.0 - pvc) * p.rho_b)
    m_pig = s * pig_share * m
    m_bind = s * (1.0 - pig_share) * m
    m_liq = (1.0 - s) * m
    m_charge = min(p.charge_frac * m, m_liq)             # liquids charged before pigments
    m_letdown = m - m_charge - m_pig                     # binder + remaining liquid
    letdown_binder_frac = m_bind / m_letdown if m_letdown > 0 else 0.0
    cp_letdown = letdown_binder_frac * p.cp_b + (1.0 - letdown_binder_frac) * p.cp_l
    vol_l = m_liq / p.rho_l + m_pig / p.rho_p + m_bind / p.rho_b
    return {"m_pigment_kg": m_pig, "m_binder_kg": m_bind, "m_liquid_kg": m_liq,
            "m_charge_kg": m_charge, "m_letdown_kg": m_letdown,
            "letdown_binder_frac": letdown_binder_frac, "cp_letdown_kj_kgk": cp_letdown,
            "pigment_share_of_solids": pig_share,
            "final_volume_m3": vol_l / 1000.0, "final_density_kg_l": m / vol_l,
            "cp_final_kj_kgk": (m_liq * p.cp_l + m_pig * p.cp_p + m_bind * p.cp_b) / m,
            "overcharged": vol_l / 1000.0 > p.working_vol_m3}


def standard_phases(recipe: Recipe, cfg: dict, *, charge_min: float | None = None,
                    pigment_min: float | None = None, dispersion_min: float | None = None,
                    letdown_min: float | None = None, charge_rpm: float | None = None,
                    pigment_rpm: float | None = None, dispersion_rpm: float | None = None,
                    letdown_rpm: float | None = None, anchor_rpm: float | None = None) -> list[Phase]:
    """The standard four-phase sequence. Defaults are the mid-points of the config ranges;
    any keyword overrides one setting (that is what the what-if simulator will do)."""
    p = PhysicsParams(cfg, recipe.family)
    comp = composition(recipe, p)

    def mid(path):
        lo, hi = get(cfg, path)
        return 0.5 * (lo + hi)

    def pick(val, path):
        return float(val) if val is not None else mid(path)

    ps = "process_steps"
    return [
        Phase("charge_liquids", 60 * pick(charge_min, f"{ps}.charge_liquids.duration_range_min"),
              pick(charge_rpm, f"{ps}.charge_liquids.hsd_rpm_range"),
              pick(anchor_rpm, f"{ps}.charge_liquids.anchor_rpm_range"),
              comp["m_charge_kg"], "liquid"),
        Phase("pigment_addition", 60 * pick(pigment_min, f"{ps}.pigment_addition.duration_range_min"),
              pick(pigment_rpm, f"{ps}.pigment_addition.hsd_rpm_range"),
              pick(anchor_rpm, f"{ps}.pigment_addition.anchor_rpm_range"),
              comp["m_pigment_kg"], "pigment"),
        Phase("high_speed_dispersion", 60 * pick(dispersion_min, f"{ps}.high_speed_dispersion.duration_range_min"),
              pick(dispersion_rpm, f"{ps}.high_speed_dispersion.hsd_rpm_range"),
              pick(anchor_rpm, f"{ps}.high_speed_dispersion.anchor_rpm_range"),
              0.0, "none"),
        Phase("letdown", 60 * pick(letdown_min, f"{ps}.letdown.duration_range_min"),
              pick(letdown_rpm, f"{ps}.letdown.hsd_rpm_range"),
              pick(anchor_rpm, f"{ps}.letdown.anchor_rpm_range"),
              comp["m_letdown_kg"], "letdown"),
    ]


# ============================================================ sub-models ==
def tip_speed_ms(rpm: float, D: float) -> float:
    return math.pi * D * rpm / 60.0


def power_number(Re: float, Kp: float, Np_turb: float) -> float:
    """Np = Kp/Re + Np_turb: laminar asymptote Kp/Re, turbulent asymptote Np_turb."""
    return Kp / Re + Np_turb if Re > 0 else NAN


def impeller_power_w(N_rps: float, rho_kg_m3: float, mu_pa_s: float, D: float,
                     Kp: float, Np_turb: float) -> tuple[float, float, float]:
    """Hydraulic shaft power [W], Reynolds number and power number of one impeller."""
    if N_rps <= 0:
        return 0.0, 0.0, NAN
    Re = rho_kg_m3 * N_rps * D ** 2 / mu_pa_s
    Np = power_number(Re, Kp, Np_turb)
    return Np * rho_kg_m3 * N_rps ** 3 * D ** 5, Re, Np


def hsd_actual_rpm(rpm_setpoint: float, rho_kg_m3: float, mu_pa_s: float, p: PhysicsParams,
                   submergence: float = 1.0, wear: float = 0.0) -> dict:
    """Actual HSD speed after the drive's kW limit. If the set-point would need more than the
    motor can give, the drive holds current and the speed sags: find the speed where
    electrical kW == limit (bisection; power rises monotonically with speed)."""
    rpm_setpoint = min(max(rpm_setpoint, 0.0), p.hsd_max_rpm)
    load_factor = submergence * (1.0 - p.wear_power_factor * wear)

    def evaluate(rpm):
        P_hyd, Re, Np = impeller_power_w(rpm / 60.0, rho_kg_m3, mu_pa_s, p.D, p.Kp, p.Np_turb)
        P_shaft = P_hyd * load_factor
        return P_shaft, P_shaft / 1000.0 / p.hsd_eff + p.hsd_no_load_kw, Re, Np

    P_shaft, kw, Re, Np = evaluate(rpm_setpoint)
    limited = kw > p.hsd_limit_kw and rpm_setpoint > 0
    rpm = rpm_setpoint
    if limited:
        lo, hi = 0.0, rpm_setpoint
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if evaluate(mid)[1] > p.hsd_limit_kw:
                hi = mid
            else:
                lo = mid
        rpm = lo
        P_shaft, kw, Re, Np = evaluate(rpm)
    return {"rpm": rpm, "shaft_w": P_shaft, "elec_kw": kw, "Re": Re, "Np": Np, "limited": limited}


def viscosity_pa_s(T_c: float, phi: float, wetted: float, p: PhysicsParams,
                   lot_factor: float = 1.0) -> tuple[float, float]:
    """Millbase viscosity at blade shear rate and the effective solids fraction used."""
    mu_vehicle = lot_factor * p.mu_ref * math.exp(-p.b * (T_c - p.T_ref))
    phi_eff = min(phi * (1.0 + p.k_agg * (1.0 - wetted)), p.phi_cap * p.phi_max)
    return mu_vehicle * (1.0 - phi_eff / p.phi_max) ** (-p.kd_n), phi_eff


def overall_U(mu_pa_s: float, anchor_rpm: float, cw_flow_m3h: float, fouling_m2k_w: float,
              p: PhysicsParams) -> dict:
    """1/U = 1/h_product + wall/k + 1/h_jacket + R_fouling, each term in m2.K/W."""
    h_j = p.h_j_ref * (max(cw_flow_m3h, 0.1) / p.flow_ref) ** p.flow_exp
    n_a = max(anchor_rpm, p.min_eff_anchor_rpm)
    h_p = p.h_p_ref * (n_a / p.anchor_ref_rpm) ** p.anchor_exp * (mu_pa_s / p.mu_ref_ht) ** p.mu_exp
    R = 1.0 / h_p + p.wall_m / p.k_wall + 1.0 / h_j + fouling_m2k_w
    return {"U": 1.0 / R, "h_product": h_p, "h_jacket": h_j}


def wetted_area_m2(V_m3: float, p: PhysicsParams) -> tuple[float, float]:
    """Jacket area in contact with product, and the liquid height (dish ignored)."""
    h = V_m3 / (math.pi / 4.0 * p.vessel_D ** 2)
    return p.A_bottom + math.pi * p.vessel_D * min(h, p.jacket_h), h


# =============================================================== stepper ==
class BatchSimulator:
    """Holds the vessel state and advances it one sample period at a time.

        sim = BatchSimulator(recipe, conditions, cfg, faults)
        sim.run_phase(phase)                       # whole phase
        sim.run_phase(phase, stop_rule=fn)         # ends early when fn(row) returns True
        df, summary = sim.finish()
    """

    def __init__(self, recipe: Recipe, cond: Conditions, cfg: dict, faults: Faults | None = None):
        self.recipe, self.cond, self.cfg = recipe, cond, cfg
        self.faults = faults or Faults()
        self.p = PhysicsParams(cfg, recipe.family)
        self.comp = composition(recipe, self.p)
        self.dt = self.p.dt
        self.H_plateau = recipe.hegman_plateau if recipe.hegman_plateau is not None else self.p.H_max
        self.e_c = self.p.e_c * recipe.grind_energy_factor
        # operator sets the blade height once per batch from the expected millbase depth
        v_mill_m3 = (self.comp["m_charge_kg"] / self.p.rho_l + self.comp["m_pigment_kg"] / self.p.rho_p) / 1000.0
        h_mill = v_mill_m3 / (math.pi / 4.0 * self.p.vessel_D ** 2)
        self.blade_h = min(max(self.p.blade_h_frac * h_mill, self.p.blade_h_min), self.p.blade_h_nominal)
        # state
        self.m_l = self.m_p = self.m_b = 0.0        # liquid, pigment, binder masses [kg]
        self.T = cond.T0_c
        self.w = 0.0                                 # wetted fraction of the pigment
        self.H = NAN                                 # model Hegman; undefined until pigment goes in
        self.E_hsd_kwh = self.E_anchor_kwh = 0.0     # electrical energy counters
        self.E_in_kj = self.E_j_kj = self.E_loss_kj = self.E_add_kj = 0.0   # balance bookkeeping
        self.t = 0.0
        self.rows: list[dict] = []
        self.phase_log: list[dict] = []

    # ---- one time step ----------------------------------------------------
    def _step(self, ph: Phase, add_step: float) -> dict:
        p, cond, comp, dt = self.p, self.cond, self.comp, self.dt

        # 1. material added during this step: mix enthalpy, update masses
        if add_step > 0:
            T_add = cond.T0_c if ph.name == "charge_liquids" else cond.T_raw_c
            cp_add = {"liquid": p.cp_l, "pigment": p.cp_p, "letdown": comp["cp_letdown_kj_kgk"]}[ph.added_kind]
            m_now = self.m_l + self.m_p + self.m_b
            if m_now > 0:
                cp_now = (self.m_l * p.cp_l + self.m_p * p.cp_p + self.m_b * p.cp_b) / m_now
                self.T = (m_now * cp_now * self.T + add_step * cp_add * T_add) / (m_now * cp_now + add_step * cp_add)
            else:
                self.T = T_add
            self.E_add_kj += add_step * cp_add * T_add
            if ph.added_kind == "liquid":
                self.m_l += add_step
            elif ph.added_kind == "pigment":
                self.m_p += add_step
            elif ph.added_kind == "letdown":
                self.m_b += add_step * comp["letdown_binder_frac"]
                self.m_l += add_step * (1.0 - comp["letdown_binder_frac"])

        # 2. bulk properties of what is in the vessel now
        m = self.m_l + self.m_p + self.m_b
        cp = (self.m_l * p.cp_l + self.m_p * p.cp_p + self.m_b * p.cp_b) / m
        vol_l = self.m_l / p.rho_l + self.m_p / p.rho_p + self.m_b / p.rho_b
        V_m3 = vol_l / 1000.0
        rho = m / V_m3
        phi = (self.m_p / p.rho_p) / vol_l
        mu, phi_eff = viscosity_pa_s(self.T, phi, self.w, p, self.recipe.viscosity_factor)
        A, h_liq = wetted_area_m2(V_m3, p)
        subm = min(max((h_liq - self.blade_h) / p.subm_ramp, 0.0), 1.0)

        # 3. power draw: HSD (with drive limit) and anchor
        hsd = hsd_actual_rpm(ph.hsd_rpm, rho, mu, p, subm, self.faults.blade_wear)
        if cond.hsd_kw_fn is not None:                       # replay: measured power overrides the model
            kw_meas = max(float(cond.hsd_kw_fn(self.t)), 0.0)
            hsd = {"rpm": float(cond.hsd_rpm_fn(self.t)) if cond.hsd_rpm_fn else hsd["rpm"],
                   "shaft_w": max(kw_meas - p.hsd_no_load_kw, 0.0) * p.hsd_eff * 1000.0,
                   "elec_kw": kw_meas, "Re": hsd["Re"], "Np": hsd["Np"], "limited": False}
        v_tip = tip_speed_ms(hsd["rpm"], p.D)
        Pa_hyd, Re_a, Np_a = impeller_power_w(ph.anchor_rpm / 60.0, rho, mu, p.Da, p.Kp_a, p.Np_turb_a)
        Pa_shaft = Pa_hyd * min(h_liq / 0.2, 1.0)            # anchor arms need ~0.2 m of liquid
        Pa_elec_kw = Pa_shaft / 1000.0 / p.anchor_eff + p.anchor_no_load_kw

        # 4. heat flows [W]; utilities may vary with time
        T_cw_in = cond.T_cw_in_fn(self.t) if cond.T_cw_in_fn else cond.T_cw_in_c
        cw_flow = cond.cw_flow_fn(self.t) if cond.cw_flow_fn else cond.cw_flow_m3h
        ht = overall_U(mu, ph.anchor_rpm, cw_flow, self.faults.jacket_fouling_m2k_w, p)
        UA = ht["U"] * A
        m_dot_cp = max(cw_flow, 0.05) * p.rho_w / 3600.0 * p.cp_w * 1000.0        # W/K
        Q_j = UA * (self.T - T_cw_in) / (1.0 + UA / (2.0 * m_dot_cp))
        T_out = T_cw_in + Q_j / m_dot_cp
        Q_loss = p.U_loss * p.A_loss * (self.T - cond.T_ambient_c) + p.k_evap * max(self.T - cond.T_ambient_c, 0.0)
        Q_in = p.eta * (hsd["shaft_w"] + Pa_shaft)
        dT = (Q_in - Q_j - Q_loss) * dt / (m * cp * 1000.0)

        # 5. wetting and grind (shaft energy per tonne of what is in the vessel)
        e_dot = hsd["shaft_w"] / 1000.0 / 3600.0 / (m / 1000.0)      # kWh/t per second
        if self.m_p > 0:
            if math.isnan(self.H):
                self.H = p.H0
            self.w += (1.0 - self.w) * e_dot * dt / p.e_wet
            f_v = max(0.0, (v_tip - p.v_min) / (p.v_ref - p.v_min)) ** p.alpha
            self.H += (self.H_plateau - self.H) * self.w * f_v * (1.0 - self.faults.blade_wear) * e_dot * dt / self.e_c

        # 6. counters and the row
        self.E_hsd_kwh += hsd["elec_kw"] * dt / 3600.0
        self.E_anchor_kwh += Pa_elec_kw * dt / 3600.0
        self.E_in_kj += Q_in * dt / 1000.0
        self.E_j_kj += Q_j * dt / 1000.0
        self.E_loss_kj += Q_loss * dt / 1000.0
        E_total = self.E_hsd_kwh + self.E_anchor_kwh
        row = {
            "t_s": self.t, "step": ph.name, "batch_mass_kg": m,
            "hsd_rpm_setpoint": ph.hsd_rpm, "hsd_rpm": hsd["rpm"], "anchor_rpm": ph.anchor_rpm,
            "hsd_kw": hsd["elec_kw"], "anchor_kw": Pa_elec_kw,
            "T_product_c": self.T, "T_jacket_in_c": T_cw_in, "T_jacket_out_c": T_out,
            "cw_flow_m3h": cw_flow, "T_ambient_c": cond.T_ambient_c,
            # internal states
            "tip_speed_ms": v_tip, "motor_limited": hsd["limited"], "hsd_shaft_kw": hsd["shaft_w"] / 1000.0,
            "blade_height_m": self.blade_h, "submergence": subm,
            "reynolds": hsd["Re"], "power_number": hsd["Np"],
            "mu_pa_s": mu, "phi": phi, "phi_eff": phi_eff, "wetted_frac": self.w,
            "density_kg_l": rho / 1000.0, "cp_kj_kgk": cp, "volume_m3": V_m3,
            "U_w_m2k": ht["U"], "A_wetted_m2": A, "Q_jacket_kw": Q_j / 1000.0, "Q_loss_kw": Q_loss / 1000.0,
            "energy_cum_kwh": E_total,
            "specific_energy_cum_kwh_t": E_total / (self.recipe.batch_mass_kg / 1000.0),
            "hegman_model": self.H,
        }
        self.rows.append(row)
        self.T += dT
        self.t += dt
        return row

    # ---- one phase --------------------------------------------------------
    def run_phase(self, ph: Phase, stop_rule: Callable[[dict], bool] | None = None) -> list[dict]:
        """Run a phase for its duration, or until stop_rule(row) returns True.
        Material is dosed at the rate mass_added / duration (an early stop leaves the
        remainder undosed, so only use stop rules on phases that add nothing)."""
        n_steps = max(int(round(ph.duration_s / self.dt)), 1)
        add_step = ph.mass_added_kg / n_steps
        t_start = self.t
        produced = []
        for _ in range(n_steps):
            row = self._step(ph, add_step)
            produced.append(row)
            if stop_rule is not None and stop_rule(row):
                break
        self.phase_log.append({"step": ph.name, "t_start_s": t_start, "t_end_s": self.t,
                               "duration_min": (self.t - t_start) / 60.0, "hsd_rpm_setpoint": ph.hsd_rpm,
                               "anchor_rpm": ph.anchor_rpm, "mass_added_kg": add_step * len(produced)})
        return produced

    # ---- wrap up ----------------------------------------------------------
    def finish(self) -> tuple[pd.DataFrame, dict]:
        df = pd.DataFrame(self.rows)
        p = self.p
        H_end_kj = (self.m_l * p.cp_l + self.m_p * p.cp_p + self.m_b * p.cp_b) * self.T
        closure_kj = (H_end_kj - self.E_add_kj) - (self.E_in_kj - self.E_j_kj - self.E_loss_kj)
        summary = batch_summary(df, self.recipe, p)
        summary.update({"energy_in_kj": self.E_in_kj, "energy_jacket_kj": self.E_j_kj,
                        "energy_loss_kj": self.E_loss_kj, "closure_error_kj": closure_kj,
                        "closure_error_pct": 100.0 * abs(closure_kj) / max(self.E_in_kj, 1e-9),
                        "overcharged": self.comp["overcharged"], "final_volume_m3": self.comp["final_volume_m3"]})
        return df, summary


def simulate_batch(recipe: Recipe, phases: list[Phase], cond: Conditions, cfg: dict,
                   faults: Faults | None = None) -> tuple[pd.DataFrame, dict]:
    """One-call wrapper: run every phase for its full duration and return (time series, summary)."""
    sim = BatchSimulator(recipe, cond, cfg, faults)
    for ph in phases:
        sim.run_phase(ph)
    return sim.finish()


def batch_summary(df: pd.DataFrame, recipe: Recipe, p: PhysicsParams) -> dict:
    """Headline numbers of one simulated batch (the same things the KPIs will score)."""
    hsd_rows = df[df["step"] == "high_speed_dispersion"]
    t_hsd0 = float(hsd_rows["t_s"].iloc[0]) if not hsd_rows.empty else NAN
    reached = df[(df["hegman_model"] >= recipe.target_hegman) & (df["t_s"] >= t_hsd0)] if not hsd_rows.empty else df.iloc[0:0]
    t_target = (float(reached["t_s"].iloc[0]) - t_hsd0) / 60.0 if not reached.empty else NAN
    peak = float(df["T_product_c"].max())
    e_total = float(df["energy_cum_kwh"].iloc[-1])
    return {
        "recipe_id": recipe.recipe_id, "family": recipe.family, "batch_mass_kg": recipe.batch_mass_kg,
        "energy_total_kwh": e_total,
        "specific_energy_kwh_per_t": e_total / (recipe.batch_mass_kg / 1000.0),
        "peak_temp_c": peak, "temp_limit_c": p.T_limit, "breached": peak > p.T_limit,
        "time_to_target_hegman_min": t_target,
        "hegman_end": float(df["hegman_model"].iloc[-1]),
        "hsd_load_peak_pct": 100.0 * float(df["hsd_kw"].max()) / p.hsd_rated_kw,
        "hsd_rpm_min_in_dispersion": float(hsd_rows["hsd_rpm"].min()) if not hsd_rows.empty else NAN,
        "motor_limited_share_pct": 100.0 * float(hsd_rows["motor_limited"].mean()) if not hsd_rows.empty else NAN,
        "dispersion_min": (float(hsd_rows["t_s"].iloc[-1]) - t_hsd0 + p.dt) / 60.0 if not hsd_rows.empty else NAN,
        "batch_cycle_min": (float(df["t_s"].iloc[-1]) + p.dt) / 60.0,
    }


def reference_recipe(family: str) -> Recipe:
    """Two reference recipes used by the Step 2 checks (typical mid-range products)."""
    if family == "water_based":
        return Recipe("water_based", 3000.0, 58.0, 55.0, 6.0, "REF-WB-MATT")
    return Recipe("solvent_based", 3000.0, 65.0, 18.0, 7.0, "REF-SB-GLOSS")
