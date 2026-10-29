"""
whatif_cli.py - run a what-if scenario or the optimiser from the command line.

Examples (from the project folder):
  python scripts/whatif_cli.py --family water_based --mass 3000 --solids 58 --pvc 55 --hegman 6 \
         --rpm 950 --anchor 35 --minutes 35 --cw-in 22 --cw-flow 10 --t-start 30 --ambient 32
  python scripts/whatif_cli.py ... --compare "rpm=1000,minutes=30" "rpm=900,cw_flow=12"
  python scripts/whatif_cli.py ... --optimise
  python scripts/whatif_cli.py ... --plot reports/my_scenario.html

Every number printed is a PREDICTION from the simulated-data twin, not plant performance.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tsd_twin import whatif as wi  # noqa: E402
from tsd_twin.config import load_config  # noqa: E402

ALIASES = {"rpm": "hsd_rpm", "anchor": "anchor_rpm", "minutes": "dispersion_min", "cw_in": "cw_inlet_c",
           "cw_flow": "cw_flow_m3h", "mass": "batch_mass_kg", "solids": "solids_pct", "pvc": "pvc_pct",
           "hegman": "target_hegman", "t_start": "T_start_c", "ambient": "T_ambient_c", "wear": "blade_wear",
           "fouling": "jacket_fouling_m2k_w"}


def parse_overrides(text: str) -> dict:
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        k = ALIASES.get(k.strip(), k.strip())
        out[k] = v.strip() == "true" if k == "stop_at_first_pass" else float(v)
    return out


def print_result(name: str, r: wi.ScenarioResult) -> None:
    sm = r.summary
    print(f"\n--- {name}: {sm['verdict']} ---")
    print(f"  energy {sm['energy_total_kwh']} kWh = {sm['specific_energy_kwh_per_t']} kWh/t | peak {sm['peak_temp_c']} degC at "
          f"{sm['peak_at_min']} min (limit {sm['temp_limit_c']}, margin {sm['margin_to_limit_c']}) | breach at {sm['breach_at_min']}")
    print(f"  time to Hegman {sm['target_hegman']}: {sm['time_to_target_hegman_min']} min | dispersion {sm['dispersion_min']} min "
          f"({sm['n_extensions']} extensions) | Hegman at end {sm['hegman_end_of_dispersion']} | cycle {sm['batch_cycle_min']} min "
          f"= {sm['throughput_t_per_h']} t/h | HSD load {sm['hsd_load_peak_pct']} % | tip {sm['tip_speed_ms']} m/s")
    for n in r.notes:
        print(f"  note: {n}")
    bad = r.checks[~r.checks["pass"]]
    for _, c in bad.iterrows():
        print(f"  {c['severity']}: {c['check']} = {c['value']} {c['unit']} (needs {c['limit']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="TSD twin what-if")
    ap.add_argument("--family", default="water_based", choices=["water_based", "solvent_based"])
    ap.add_argument("--mass", type=float, default=3000)
    ap.add_argument("--solids", type=float, default=58)
    ap.add_argument("--pvc", type=float, default=55)
    ap.add_argument("--hegman", type=float, default=6.0)
    ap.add_argument("--rpm", type=float, default=950)
    ap.add_argument("--anchor", type=float, default=35)
    ap.add_argument("--minutes", type=float, default=35, help="planned high-speed dispersion time")
    ap.add_argument("--cw-in", type=float, default=22)
    ap.add_argument("--cw-flow", type=float, default=10)
    ap.add_argument("--t-start", type=float, default=30)
    ap.add_argument("--ambient", type=float, default=32)
    ap.add_argument("--stop-at-first-pass", action="store_true")
    ap.add_argument("--compare", nargs="*", default=[], help='overrides like "rpm=1000,minutes=30"')
    ap.add_argument("--optimise", action="store_true")
    ap.add_argument("--plot", type=str, default=None, help="write an HTML plot of the scenarios")
    a = ap.parse_args()

    cfg = load_config()
    twin = wi.load_twin(ROOT / "models")
    base = wi.Settings(a.family, a.mass, a.solids, a.pvc, a.hegman, a.rpm, a.anchor, a.minutes, a.cw_in, a.cw_flow,
                       a.t_start, a.ambient, stop_at_first_pass=a.stop_at_first_pass)
    print(f"twin: residual model {'loaded' if twin.temperature else 'NOT loaded'}; energy factor {twin.energy_factor:.3f}, "
          f"grind-time factor {twin.grind_time_factor:.3f} ({twin.n_calibration_batches} calibration batches)")
    results = {"base": wi.run_scenario(base, cfg, twin)}
    print_result("base", results["base"])
    for i, text in enumerate(a.compare, 1):
        s = replace(base, **parse_overrides(text))
        results[f"scenario {i} ({text})"] = wi.run_scenario(s, cfg, twin)
        print_result(f"scenario {i} ({text})", results[f"scenario {i} ({text})"])
    if len(results) > 1:
        pd.set_option("display.width", 200)
        print("\n" + wi.compare_scenarios(results).to_string())
    if a.optimise:
        opt = wi.optimise(base, cfg, twin)
        print(f"\n=== OPTIMISER: {opt['status']} ===")
        print(opt["explanation"])
        if opt["best"] is not None:
            print_result("optimum", opt["best"])
            results["optimum"] = opt["best"]
    if a.plot:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly not installed - no plot")
            return
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Product temperature", "Grind (Hegman)", "Cumulative energy"))
        colours = ["#1F3864", "#C55A11", "#548235", "#7030A0", "#C00000"]
        for (name, r), col in zip(results.items(), colours):
            c = r.curve
            fig.add_trace(go.Scatter(x=c["t_min"], y=c["T_pred_c"], name=name, line=dict(color=col, width=2)), 1, 1)
            fig.add_trace(go.Scatter(x=c["t_min"], y=c["hegman"], name=name, line=dict(color=col, width=2), showlegend=False), 1, 2)
            fig.add_trace(go.Scatter(x=c["t_min"], y=c["energy_cum_kwh"], name=name, line=dict(color=col, width=2), showlegend=False), 1, 3)
        lim = results["base"].summary["temp_limit_c"]
        fig.add_hline(y=lim, line=dict(color="#C00000", dash="dot"), row=1, col=1, annotation_text=f"{lim} degC limit")
        fig.add_hline(y=a.hegman, line=dict(color="grey", dash="dot"), row=1, col=2, annotation_text="target")
        fig.update_layout(template="plotly_white", font=dict(family="Arial"), height=420,
                          title="What-if scenarios (predictions from the simulated-data twin)")
        fig.write_html(a.plot, include_plotlyjs=True)
        print(f"\nplot written to {a.plot}")


if __name__ == "__main__":
    main()
