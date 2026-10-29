"""
check_step2.py - proves the physics model behaves like a disperser before any data is simulated.

What it does:
  1. loads the config and lists the physics parameters (basis = given / assumption / physical)
  2. static curves: power number vs Reynolds, viscosity vs temperature / wetting / solids
  3. power vs rpm at a fixed millbase, showing where the 55 kW drive limit bites
  4. runs two reference batches (water-based matt, solvent-based gloss) and prints phase tables
  5. energy-balance closure: stored heat == heat in - jacket - losses (must close to ~0)
  6. sensitivity checks: each disturbance must move the result in the physically right direction
  7. writes an interactive plot of the two reference batches to reports/step2_reference_batches.html

Run from the project folder:   python scripts/check_step2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import physics as ph  # noqa: E402
from tsd_twin.config import get, load_config, parameter_table  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 200)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def phase_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("step", sort=False)
    return g.agg(min=("t_s", lambda s: round((s.iloc[-1] - s.iloc[0] + 10) / 60, 1)),
                 mass_kg=("batch_mass_kg", "last"),
                 T_start=("T_product_c", "first"), T_end=("T_product_c", "last"),
                 rpm_sp=("hsd_rpm_setpoint", "first"), rpm_min=("hsd_rpm", "min"),
                 kw_mean=("hsd_kw", "mean"), kw_max=("hsd_kw", "max"),
                 mu_start=("mu_pa_s", "first"), mu_end=("mu_pa_s", "last"),
                 Re_mean=("reynolds", "mean"), U=("U_w_m2k", "mean"), A_m2=("A_wetted_m2", "mean"),
                 Qj_kw=("Q_jacket_kw", "mean"), wetted=("wetted_frac", "last"),
                 hegman=("hegman_model", "last"), kwh_t=("specific_energy_cum_kwh_t", "last")).round(2)


def main() -> None:
    cfg = load_config()
    (ROOT / "reports").mkdir(exist_ok=True)

    # ------------------------------------------------------------------- 1
    banner("1. PHYSICS PARAMETERS (section 'physics' of the one assumptions file)")
    table = parameter_table(cfg)
    phys = table[table["parameter"].str.startswith("physics.")]
    print(phys["basis"].value_counts().rename("count").to_string())
    print(phys[["parameter", "value", "unit", "basis"]].to_string(index=False))
    phys.to_csv(ROOT / "reports" / "step2_physics_register.csv", index=False)

    # ------------------------------------------------------------------- 2
    banner("2. STATIC CURVES")
    Kp, Npt = get(cfg, "physics.hsd_power.laminar_constant_kp"), get(cfg, "physics.hsd_power.np_turbulent")
    print("Power number vs Reynolds (HSD):")
    for Re in [10, 100, 1000, 3000, 10000, 100000]:
        print(f"  Re = {Re:>7,d}   Np = {ph.power_number(Re, Kp, Npt):6.3f}")
    print("  -> below Re ~ 1,000 the blade is in thick product and Np rises sharply (laminar);")
    print("     above Re ~ 10,000 Np flattens at the turbulent value. Paint millbase sits in between.")

    for fam in ["water_based", "solvent_based"]:
        p = ph.PhysicsParams(cfg, fam)
        r = ph.reference_recipe(fam)
        comp = ph.composition(r, p)
        phi_mill = (comp["m_pigment_kg"] / p.rho_p) / (comp["m_pigment_kg"] / p.rho_p + comp["m_charge_kg"] / p.rho_l)
        print(f"\nViscosity at blade shear rate, {fam} millbase (phi = {phi_mill:.3f}):")
        print("   T degC   unwetted   wetted   [Pa.s]")
        for T in ([25, 35, 45] if fam == "water_based" else [25, 40, 55]):
            mu_u, _ = ph.viscosity_pa_s(T, phi_mill, 0.0, p)
            mu_w, _ = ph.viscosity_pa_s(T, phi_mill, 1.0, p)
            print(f"   {T:5.0f}     {mu_u:6.2f}    {mu_w:6.2f}")
        print("   solids effect at 25 degC, wetted:  " + "  ".join(
            f"phi={x:.2f}: {ph.viscosity_pa_s(25, x, 1.0, p)[0]:.2f}" for x in [0.0, 0.2, 0.3, 0.4, 0.5]))

    # ------------------------------------------------------------------- 3
    banner("3. HSD POWER vs SPEED  (water-based millbase, rho 1.70 kg/L, mu 1.0 Pa.s)")
    p = ph.PhysicsParams(cfg, "water_based")
    print("  set rpm  actual rpm  tip m/s   shaft kW  elec kW  load %  limited")
    for rpm in [500, 764, 912, 1000, 1061, 1200, 1450]:
        h = ph.hsd_actual_rpm(rpm, 1700.0, 1.0, p)
        print(f"  {rpm:7.0f}  {h['rpm']:10.0f}  {ph.tip_speed_ms(h['rpm'], p.D):7.1f}  "
              f"{h['shaft_w'] / 1000:9.1f}  {h['elec_kw']:7.1f}  {100 * h['elec_kw'] / p.hsd_rated_kw:6.0f}  {h['limited']}")
    print("  -> above ~1,050 rpm the drive is at its kW limit: the set-point is not reached in this product.")

    # ------------------------------------------------------------------- 4
    banner("4. REFERENCE BATCHES (physics only, 10 s steps; SIMULATED, not plant data)")
    runs = {}
    for fam in ["water_based", "solvent_based"]:
        r = ph.reference_recipe(fam)
        pp = ph.PhysicsParams(cfg, fam)
        comp = ph.composition(r, pp)
        df, s = ph.simulate_batch(r, ph.standard_phases(r, cfg), ph.Conditions(), cfg)
        runs[fam] = (df, s)
        df.to_csv(ROOT / "reports" / f"step2_reference_{fam}.csv", index=False)
        print(f"\n{r.recipe_id}: {r.batch_mass_kg:.0f} kg, solids {r.solids_pct} %, PVC {r.pvc_pct} %, "
              f"target Hegman {r.target_hegman}")
        print(f"  composition: pigment {comp['m_pigment_kg']:.0f} kg, binder {comp['m_binder_kg']:.0f} kg, "
              f"liquid {comp['m_liquid_kg']:.0f} kg; charge {comp['m_charge_kg']:.0f}, letdown {comp['m_letdown_kg']:.0f}; "
              f"final cp {comp['cp_final_kj_kgk']:.2f} kJ/kgK, density {comp['final_density_kg_l']:.2f} kg/L, "
              f"volume {comp['final_volume_m3']:.2f} m3")
        print(phase_table(df).to_string())
        print(f"  energy {s['energy_total_kwh']:.1f} kWh = {s['specific_energy_kwh_per_t']:.1f} kWh/t | "
              f"peak T {s['peak_temp_c']:.1f} degC (limit {s['temp_limit_c']}) breach={s['breached']} | "
              f"Hegman {r.target_hegman} after {s['time_to_target_hegman_min']:.1f} min of HSD, end {s['hegman_end']:.2f} | "
              f"peak load {s['hsd_load_peak_pct']:.0f} %")

    # ------------------------------------------------------------------- 5
    banner("5. ENERGY-BALANCE CLOSURE")
    for fam, (df, s) in runs.items():
        print(f"  {fam:14s} heat in {s['energy_in_kj'] / 3600:7.2f} kWh | jacket {s['energy_jacket_kj'] / 3600:7.2f} kWh | "
              f"losses {s['energy_loss_kj'] / 3600:6.2f} kWh | closure error {s['closure_error_kj']:.3f} kJ "
              f"({s['closure_error_pct']:.4f} %)")
        assert s["closure_error_pct"] < 0.01, "energy balance does not close"
    print("  -> stored heat = heat in - jacket - losses, to numerical precision.  OK")

    # ------------------------------------------------------------------- 6
    banner("6. SENSITIVITY: does each disturbance move the result the right way?")
    base_r = ph.reference_recipe("water_based")
    base_df, base = ph.simulate_batch(base_r, ph.standard_phases(base_r, cfg), ph.Conditions(), cfg)
    cases = [
        ("baseline", base_r, {}, ph.Conditions(), ph.Faults()),
        ("CW inlet 28 degC (summer)", base_r, {}, ph.Conditions(T_cw_in_c=28.0, T0_c=33.0, T_raw_c=33.0, T_ambient_c=35.0), ph.Faults()),
        ("CW flow 5 m3/h (low flow)", base_r, {}, ph.Conditions(cw_flow_m3h=5.0), ph.Faults()),
        ("jacket fouled 0.001 m2K/W", base_r, {}, ph.Conditions(), ph.Faults(jacket_fouling_m2k_w=0.001)),
        ("HSD 1061 rpm (25 m/s)", base_r, {"dispersion_rpm": 1061}, ph.Conditions(), ph.Faults()),
        ("blade wear 30 %", base_r, {}, ph.Conditions(), ph.Faults(blade_wear=0.3)),
        ("batch 4,500 kg (overcharge)", ph.Recipe("water_based", 4500, 58, 55, 6.0, "WB-4500"), {}, ph.Conditions(), ph.Faults()),
        ("dispersion 60 min, summer", base_r, {"dispersion_min": 60}, ph.Conditions(T_cw_in_c=28.0, T0_c=33.0, T_raw_c=33.0, T_ambient_c=35.0), ph.Faults()),
        ("summer + 1000 rpm + 7 m3/h, 50 min", base_r, {"dispersion_min": 50, "dispersion_rpm": 1000},
         ph.Conditions(T_cw_in_c=28.0, T0_c=33.0, T_raw_c=33.0, T_ambient_c=35.0, cw_flow_m3h=7.0), ph.Faults()),
    ]
    rows = []
    results = {}
    for name, r, over, cond, flt in cases:
        df, s = ph.simulate_batch(r, ph.standard_phases(r, cfg, **over), cond, cfg, flt)
        results[name] = s
        hsd = df[df["step"] == "high_speed_dispersion"]
        rows.append({"case": name, "peak_T_c": s["peak_temp_c"], "breach": s["breached"],
                     "kWh": s["energy_total_kwh"], "kWh_per_t": s["specific_energy_kwh_per_t"],
                     "t_to_Hegman_min": s["time_to_target_hegman_min"], "Hegman_end": s["hegman_end"],
                     "HSD_kw_mean": hsd["hsd_kw"].mean(), "rpm_min": s["hsd_rpm_min_in_dispersion"],
                     "limited_%": s["motor_limited_share_pct"], "overcharged": s["overcharged"]})
    sens = pd.DataFrame(rows).round(2)
    print(sens.to_string(index=False))
    sens.to_csv(ROOT / "reports" / "step2_sensitivity.csv", index=False)

    b = results["baseline"]
    assert results["CW inlet 28 degC (summer)"]["peak_temp_c"] > b["peak_temp_c"], "warm CW must raise peak T"
    assert results["CW flow 5 m3/h (low flow)"]["peak_temp_c"] > b["peak_temp_c"], "low flow must raise peak T"
    assert results["jacket fouled 0.001 m2K/W"]["peak_temp_c"] > b["peak_temp_c"], "fouling must raise peak T"
    assert results["HSD 1061 rpm (25 m/s)"]["time_to_target_hegman_min"] < b["time_to_target_hegman_min"], "faster blade must grind sooner"
    assert results["HSD 1061 rpm (25 m/s)"]["motor_limited_share_pct"] > 0, "1061 rpm must hit the drive limit"
    wear = results["blade wear 30 %"]
    assert (np.isnan(wear["time_to_target_hegman_min"]) or wear["time_to_target_hegman_min"] > b["time_to_target_hegman_min"]), "worn blade must grind slower"
    assert wear["energy_total_kwh"] < b["energy_total_kwh"], "worn blade must draw less power"
    assert results["batch 4,500 kg (overcharge)"]["overcharged"], "4,500 kg water-based must overcharge the vessel"
    assert results["summer + 1000 rpm + 7 m3/h, 50 min"]["breached"], "adverse summer case should breach 45 degC"
    print("\n  all directional checks OK")

    # ------------------------------------------------------------------- 7
    banner("7. PLOT")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed (pip install plotly) - plot skipped, everything else passed.")
        banner("ALL STEP 2 CHECKS PASSED - ready for Step 3 (simulated dataset)")
        return

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12, horizontal_spacing=0.08,
                        subplot_titles=("Product and jacket temperature", "HSD speed and electrical power",
                                        "Millbase viscosity at blade shear rate", "Grind (model Hegman) and specific energy"),
                        specs=[[{"secondary_y": False}, {"secondary_y": True}],
                               [{"secondary_y": False}, {"secondary_y": True}]])
    colours = {"water_based": "#1F3864", "solvent_based": "#C55A11"}
    for fam, (df, s) in runs.items():
        x = df["t_s"] / 60.0
        c = colours[fam]
        lab = "WB matt" if fam == "water_based" else "SB gloss"
        fig.add_trace(go.Scatter(x=x, y=df["T_product_c"], name=f"{lab} product T", line=dict(color=c, width=2.2)), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=df["T_jacket_out_c"], name=f"{lab} jacket out", line=dict(color=c, width=1, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=df["hsd_kw"], name=f"{lab} HSD kW", line=dict(color=c, width=2.2)), 1, 2, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=df["hsd_rpm"], name=f"{lab} HSD rpm", line=dict(color=c, width=1, dash="dash")), 1, 2, secondary_y=True)
        fig.add_trace(go.Scatter(x=x, y=df["mu_pa_s"], name=f"{lab} viscosity", line=dict(color=c, width=2.2)), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=df["hegman_model"], name=f"{lab} Hegman", line=dict(color=c, width=2.2)), 2, 2, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=df["specific_energy_cum_kwh_t"], name=f"{lab} kWh/t", line=dict(color=c, width=1, dash="dash")), 2, 2, secondary_y=True)
    for fam, lim in [("water_based", 45), ("solvent_based", 55)]:
        fig.add_hline(y=lim, line=dict(color=colours[fam], dash="dot", width=1), row=1, col=1,
                      annotation_text=f"{lim} degC limit", annotation_position="top left")
    fig.add_hline(y=55, line=dict(color="grey", dash="dot", width=1), row=1, col=2, annotation_text="55 kW motor")
    # phase shading from the water-based run
    df0 = runs["water_based"][0]
    for step, grp in df0.groupby("step", sort=False):
        fig.add_vrect(x0=grp["t_s"].iloc[0] / 60, x1=(grp["t_s"].iloc[-1] + 10) / 60, fillcolor="grey",
                      opacity=0.06 if step in ("charge_liquids", "high_speed_dispersion") else 0.0, line_width=0,
                      annotation_text=step.replace("_", " "), annotation_position="bottom left", annotation_font_size=10, row=1, col=1)
    fig.update_layout(title="TSD physics model - two reference batches (simulated, not plant data)",
                      template="plotly_white", height=760, legend=dict(orientation="h", y=-0.08), font=dict(family="Arial"))
    fig.update_xaxes(title_text="minutes from batch start", row=2, col=1)
    fig.update_xaxes(title_text="minutes from batch start", row=2, col=2)
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="rpm", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Pa.s", row=2, col=1)
    fig.update_yaxes(title_text="Hegman", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="kWh/t", row=2, col=2, secondary_y=True)
    out = ROOT / "reports" / "step2_reference_batches.html"
    fig.write_html(out, include_plotlyjs=True)      # self-contained: opens offline on the plant network
    print(f"  written {out.relative_to(ROOT)}  (open in a browser)")

    banner("ALL STEP 2 CHECKS PASSED - ready for Step 3 (simulated dataset)")


if __name__ == "__main__":
    main()
