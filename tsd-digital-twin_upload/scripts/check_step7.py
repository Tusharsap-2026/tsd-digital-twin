"""
check_step7.py - precomputes the runtime results for the whole history, then starts the dashboard
headless and screenshots every page to prove it renders.

What it does:
  1. runtime baselines from the TRAIN batches -> models/runtime_calibration.json
  2. runs the twin over every batch -> data/processed/fleet_verdicts.csv, fleet_alerts.csv
     and prints the first-pass alert statistics (tuned properly in Step 8)
  3. launches `streamlit run dashboard/app.py` on a free port, opens each of the four pages with
     Playwright, saves reports/step7_page_*.png and checks for Streamlit error boxes

Run from the project folder:   python scripts/check_step7.py
Then open the dashboard yourself:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = Path(os.environ.get("TSD_DATA_DIR", ROOT / "data" / "raw"))
_PROC_DIR = ROOT / "data" / ("processed" if _RAW_DIR.name == "raw" else f"processed_{_RAW_DIR.name}")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import models as md, runtime as rt  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

pd.set_option("display.width", 200)
RAW, PROC, MODELS, REPORTS = _RAW_DIR, _PROC_DIR, ROOT / "models", ROOT / "reports"


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main(screenshots: bool = True) -> None:
    cfg = load_config()
    REPORTS.mkdir(exist_ok=True)
    tm, bm, card = md.load_models(MODELS)
    ts = pd.read_csv(RAW / "timeseries.csv", parse_dates=["timestamp"])
    batches = pd.read_csv(RAW / "batches.csv", parse_dates=["start_time"]).set_index("batch_id")
    split = pd.read_csv(PROC / "split.csv").set_index("batch_id")
    thr = rt.Thresholds.from_config(cfg)

    # ------------------------------------------------------------------- 1
    banner("1. RUNTIME BASELINES (the normal plant-vs-physics gap, from the TRAIN batches)")
    tr = split[split["split_random"] == "train"].index
    t = time.time()
    rows = []
    for bid in tr:
        run = rt.prepare_batch(ts[ts["batch_id"] == bid], cfg, tm, float(batches.loc[bid, "batch_mass_nominal_kg"]),
                               float(batches.loc[bid, "cw_flow_setpoint_m3h"]), thr)
        rows.append(rt.batch_verdict(run))
    Vtr = pd.DataFrame(rows)
    bl = rt.Baselines(kw_resid_pct=float(Vtr["kw_resid_pct_hsd_raw"].median()), T_resid_replay_c=float(Vtr["T_resid_replay_hsd_raw"].median()),
                      u_ratio=float(Vtr["u_ratio_hsd_raw"].median()), n_batches=len(Vtr))
    bl.save(MODELS / "runtime_calibration.json", "medians over the train batches of the SIMULATED dataset")
    print(f"  HSD power runs {bl.kw_resid_pct:+.1f} % above the physics on a normal batch; the product runs {bl.T_resid_replay_c:+.1f} degC "
          f"above the measured-power physics; the jacket transfers {100 * bl.u_ratio:.0f} % of the physics U  ({len(tr)} batches, {time.time() - t:.0f} s)")
    print("  -> fault rules look for departures FROM these levels, not from the textbook physics")

    # ------------------------------------------------------------------- 2
    banner("2. RUN THE TWIN OVER EVERY BATCH (what the dashboard shows)")
    t = time.time()
    verdicts, alerts = [], []
    for bid in batches.index:
        run = rt.prepare_batch(ts[ts["batch_id"] == bid], cfg, tm, float(batches.loc[bid, "batch_mass_nominal_kg"]),
                               float(batches.loc[bid, "cw_flow_setpoint_m3h"]), thr, bl)
        verdicts.append(rt.batch_verdict(run))
        alerts.append(rt.alert_table(run))
    V = pd.DataFrame(verdicts).set_index("batch_id").join(batches[["start_time", "recipe_id", "recipe_name", "product_family", "batch_mass_kg",
                                                                    "dispersion_min", "n_extensions", "hegman_reached_lab", "fault_injected"]]).reset_index()
    A = pd.concat([a for a in alerts if not a.empty], ignore_index=True)
    V.to_csv(PROC / "fleet_verdicts.csv", index=False)
    A.to_csv(PROC / "fleet_alerts.csv", index=False)
    print(f"  {len(V)} batches in {time.time() - t:.0f} s -> data/processed/fleet_verdicts.csv ({len(A)} alert episodes -> fleet_alerts.csv)")
    br = V[V["breached"]]
    print(f"  temperature: {len(br)} breaches; pre-alarm raised before the crossing on {int(br['prealarm_lead_min'].notna().sum())} of them, "
          f"lead time median {br['prealarm_lead_min'].median():.0f} min (min {br['prealarm_lead_min'].min():.0f}); "
          f"{int(((V['temp_flag'] == 'PREALARM') & ~V['breached']).sum())} watch flags on batches that stayed under the limit")
    assert br["prealarm_lead_min"].notna().all(), "a breach had no pre-alarm"
    assert br["prealarm_lead_min"].min() >= get(cfg, "kpi_targets.breach_lead_time_min"), "a pre-alarm came too late"
    print("  machine flags vs the simulator's hidden faults (thresholds from config; Step 8 tunes them):")
    act = batches.loc[V["batch_id"], "faults_active"].fillna("").values
    wear_t, foul_t = batches.loc[V["batch_id"], "blade_wear"].values, batches.loc[V["batch_id"], "jacket_fouling_m2k_w"].values
    for key, name, clear in [("BLADE", "BLADE_WEAR", wear_t < 0.10), ("FOUL", "JACKET_FOULING", foul_t < 0.0003),
                             ("FLOW", "LOW_CW_FLOW", None), ("OVER", "OVERCHARGE", None)]:
        a = np.array([name in s for s in act])
        f = V["flags"].str.contains(key).values
        clear = ~a if clear is None else clear
        grey = ~a & ~clear
        print(f"    {name:15s} caught {int((a & f).sum()):3d} of {int(a.sum()):3d}  ({100 * (a & f).sum() / max(a.sum(), 1):3.0f} %),  "
              f"false flags {int((clear & f).sum()):3d} of {int(clear.sum()):3d} clearly healthy"
              + (f", early flags {int((grey & f).sum()):3d} of {int(grey.sum()):3d} in the grey zone" if grey.any() else ""))
    print(f"  twin accuracy over the year: now-cast MAE median {V['nowcast_mae'].median():.2f} degC, 5-min-ahead {V['ahead_mae'].median():.2f} degC")

    # ------------------------------------------------------------------- 3
    if not screenshots:
        banner("STEP 7 RUNTIME DONE (screenshots skipped)")
        return
    banner("3. DASHBOARD SMOKE TEST (headless Streamlit + Playwright screenshots)")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  playwright not installed - start the dashboard yourself:  streamlit run dashboard/app.py")
        banner("STEP 7 DONE")
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, STREAMLIT_SERVER_HEADLESS="true", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"), "--server.port", str(port),
                             "--server.headless", "true", "--server.address", "127.0.0.1"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            time.sleep(1)
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
        pages = [("live_batch", ""), ("what_if", "whatif"), ("batch_history", "history"), ("alerts", "alerts")]
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 2600})
            for name, path in pages:
                page.goto(f"http://127.0.0.1:{port}/{path}", wait_until="networkidle", timeout=120000)
                for _ in range(90):               # wait until Streamlit has finished running the script
                    time.sleep(1)
                    running = page.locator('[data-testid="stStatusWidget"]').count()
                    if running == 0 and page.locator('.js-plotly-plot').count() > 0:
                        break
                time.sleep(2)
                errors = page.locator('[data-testid="stException"]').count()
                plots = page.locator('.js-plotly-plot').count()
                out = REPORTS / f"step7_page_{name}.png"
                page.screenshot(path=str(out), full_page=True)   # full page: the main container scrolls, so grow the viewport first
                print(f"  {name:14s} plots {plots}  exceptions {errors}  -> {out.relative_to(ROOT)}")
                assert errors == 0, f"page {name} shows a Streamlit exception"
                assert plots > 0, f"page {name} rendered no chart"
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    banner("ALL STEP 7 CHECKS PASSED - open the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main(screenshots="--no-screenshots" not in sys.argv)
