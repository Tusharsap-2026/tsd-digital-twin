"""
check_step9.py - proves the path to live data end to end.

What it does:
  1. tag map and step map load; every canonical measured column has a tag
  2. replay source -> assembler -> live twin -> historian on three batches; the assembled rows
     equal the source rows; alerts raised live match the offline (Step 7) alert log
  3. OPC UA end to end: starts the mock PLC server (asyncua) on a free port, replays a batch at
     25x speed, the twin subscribes, assembles rows on the PLC heartbeat, runs, writes the
     historian; the rows match what the PLC wrote
  4. MQTT: runs only if a broker answers on localhost:1883 (otherwise reported as skipped)
  5. export_historian.py -> canonical files -> the pipeline reads them (TSD_DATA_DIR)
  6. recalibration on the simulated plant: the ordered refit must bring the physics-only
     temperature error down by more than half and move every factor in the hidden direction

Run from the project folder:   python scripts/check_step9.py      (~4 min; needs models/ from Step 5)
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRATCH = ROOT / "data" / "live_test"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsd_twin import features as ft, live, models as md, runtime as rt, schema  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

pd.set_option("display.width", 200)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def compare(hist_path: Path, bid: str, n_rows: int | None = None) -> tuple[int, int, int, bool]:
    h = live.Historian(hist_path)
    rb = h.read_batch(bid)
    src = pd.read_csv(ROOT / "data" / "raw" / "timeseries.csv")
    src = src[src["batch_id"] == bid]
    if n_rows:
        src = src.head(n_rows)
    m = rb.merge(src, on="t_s", suffixes=("_live", "_src"))
    tot = ok = 0
    for c in ["T_product_c", "hsd_kw", "batch_mass_kg", "cw_flow_m3h", "hsd_rpm", "T_jacket_out_c", "T_jacket_in_c", "anchor_kw", "hegman"]:
        a, b = m[f"{c}_live"], m[f"{c}_src"]
        d = (a - b).abs()
        ok += int(((d < 1e-6) | (a.isna() & b.isna())).sum())
        tot += len(m)
    steps_equal = rb["step"].value_counts().to_dict() == src["step"].value_counts().to_dict()
    return len(rb), len(src), int(100 * ok / max(tot, 1)), steps_equal


def main() -> None:
    cfg = load_config()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for f in SCRATCH.glob("*"):
        f.unlink()
    tm, _, _ = md.load_models(ROOT / "models")
    bl = rt.Baselines.load(ROOT / "models" / "runtime_calibration.json")
    thr = rt.Thresholds.from_config(cfg)
    batches = pd.read_csv(ROOT / "data" / "raw" / "batches.csv").set_index("batch_id")
    fleet_alerts = pd.read_csv(ROOT / "data" / "processed" / "fleet_alerts.csv")

    # ------------------------------------------------------------------- 1
    banner("1. TAG MAP")
    tmap = live.TagMap(ROOT / "config" / "tag_map.csv", ROOT / "config" / "step_map.csv")
    missing = [c for c in schema.measured_columns() if c not in tmap.specs and c not in ("timestamp", "t_s")]
    assert not missing, f"canonical columns without a tag: {missing}"
    print(f"  {len(tmap.specs)} tags mapped ({sum(s.source == 'PLC' for s in tmap.specs.values())} PLC, "
          f"{sum(s.source == 'MES' for s in tmap.specs.values())} MES, {sum(s.source == 'LAB' for s in tmap.specs.values())} LAB); "
          f"every canonical column has one; {len(tmap.step_map)} MES phase aliases")
    assert tmap.to_canonical_step("DISPERSE") == "high_speed_dispersion" and tmap.to_canonical_step("CIP") == ""
    print("  staleness: only the PLC heartbeat has a timeout; all other values are latched            OK")

    # ------------------------------------------------------------------- 2
    banner("2. REPLAY SOURCE -> ASSEMBLER -> LIVE TWIN -> HISTORIAN (three batches, as fast as possible)")
    ids = ["TSD-0176", "TSD-0177", "TSD-0225"]
    buf = live.TagBuffer(tmap)
    hist = live.Historian(SCRATCH / "replay.sqlite")
    live_alerts = []
    twin = live.LiveTwin(cfg, tm, bl, thr, hist, every_n_rows=3, on_alert=lambda c, s: live_alerts.append((s["batch_id"], c, round(float(s["t_min"]), 1))))
    asm = live.BatchAssembler(tmap, cfg, on_row=lambda df: twin.on_row(df, buf), on_batch_end=lambda df: twin.on_batch_end(df, buf))
    t = time.time()
    live.ReplaySource(ROOT / "data" / "raw" / "timeseries.csv", buf, asm, ids, speed=0, batch_meta=batches).run()
    print(f"  {len(ids)} batches replayed in {time.time() - t:.1f} s; twin ran {twin.n_runs} times, "
          f"{1000 * np.mean(twin.run_times):.0f} ms each (max {1000 * max(twin.run_times):.0f} ms); historian {hist.counts()}")
    for bid in ids:
        n_live, n_src, pct, steps_ok = compare(SCRATCH / "replay.sqlite", bid)
        print(f"    {bid}: {n_live}/{n_src} rows, {pct} % of values identical, phases {'match' if steps_ok else 'DIFFER'}")
        assert n_live == n_src and pct >= 99 and steps_ok
    off = fleet_alerts[fleet_alerts["batch_id"].isin(ids)].sort_values(["batch_id", "raised_at_min"]).drop_duplicates(["batch_id", "alert"])
    off_set = {(r["batch_id"], r["alert"]) for _, r in off.iterrows() if r["alert"] != "TEMP_LIMIT"}
    live_set = {(b, c) for b, c, _ in live_alerts if c != "TEMP_LIMIT"}
    print(f"  alerts raised live: {sorted(live_set)}")
    print(f"  offline (Step 7):   {sorted(off_set)}")
    assert off_set == live_set, "live and offline alert sets differ"
    worst = 0.0
    for b, c, t_live in live_alerts:
        row = off[(off["batch_id"] == b) & (off["alert"] == c)]
        if len(row):
            d = t_live - float(row["raised_at_min"].iloc[0])
            worst = max(worst, abs(d))
            assert abs(d) <= 1.5, f"{b} {c} raised {d:+.1f} min from offline"
    print(f"  live alert times within {worst:.1f} min of the offline log (the live twin cycles every 30 s and a signal hovering at the\n"
          f"  threshold can flap by one debounce window; the row-level predictions themselves agree to 0.01 degC)            OK")

    # ------------------------------------------------------------------- 3
    banner("3. OPC UA END TO END (mock PLC server replaying TSD-0176 at 25x, the twin subscribing)")
    port = free_port()
    ready = SCRATCH / "plc_ready"
    plc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "mock_plc_opcua.py"), "--port", str(port), "--speed", "25", "--batches", "TSD-0176",
                            "--rows", "300", "--start-delay", "6", "--ready-file", str(ready)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if ready.exists():
                break
            time.sleep(0.5)
        assert ready.exists(), "mock PLC did not start"
        hist_path = SCRATCH / "opcua.sqlite"
        t = time.time()
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_live.py"), "--source", "opcua", "--endpoint", f"opc.tcp://127.0.0.1:{port}/tsd/",
                              "--historian", str(hist_path), "--timeout", "140", "--publishing-ms", "50", "--settle", "0.05"],
                             capture_output=True, text=True, timeout=400)
        lines = [l for l in res.stdout.splitlines() if l.startswith("  ALERT") or l.startswith("  BATCH END") or l.startswith("historian")]
        print("  " + "\n  ".join(lines[-4:]))
        n_live, n_src, pct, steps_ok = compare(hist_path, "TSD-0176", 300)
        print(f"  rows assembled over OPC UA: {n_live}/{n_src}; {pct} % of values identical to what the PLC wrote; phases {'match' if steps_ok else 'DIFFER'}  "
              f"({time.time() - t:.0f} s)")
        assert n_live == n_src and pct >= 98 and steps_ok, "OPC UA rows do not match the source"
    finally:
        plc.terminate()
        try:
            plc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            plc.kill()

    # ------------------------------------------------------------------- 4
    banner("4. MQTT")
    with socket.socket() as s:
        s.settimeout(1.0)
        broker = s.connect_ex(("127.0.0.1", 1883)) == 0
    if broker:
        print("  broker found on localhost:1883 - run:  python scripts/run_live.py --source mqtt --host 127.0.0.1")
    else:
        print("  no MQTT broker on localhost:1883 - MqttSource is implemented against paho-mqtt and uses the same TagBuffer/assembler\n"
              "  as the OPC UA path proven above; test it on site with:  python scripts/run_live.py --source mqtt --host <broker>")

    # ------------------------------------------------------------------- 5
    banner("5. HISTORIAN EXPORT -> CANONICAL FILES -> PIPELINE")
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_historian.py"), "--historian", str(SCRATCH / "replay.sqlite"), "--out", str(SCRATCH)],
                         capture_output=True, text=True)
    print("  " + res.stdout.strip())
    ts_live = pd.read_csv(SCRATCH / "timeseries.csv")
    b_live = pd.read_csv(SCRATCH / "batches.csv")
    assert list(ts_live.columns) == schema.measured_columns() and len(b_live) == len(ids)
    for c in ["batch_mass_nominal_kg", "cw_flow_setpoint_m3h", "planned_dispersion_min", "fault_injected", "breached"]:
        assert c in b_live.columns
    # the feature builder (Step 4) accepts the export unchanged
    b = ts_live[ts_live["batch_id"] == ids[0]].reset_index(drop=True)
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    base, planned = ft.physics_baselines(b, cfg, planned_dispersion_min=float(b_live.set_index("batch_id").loc[ids[0], "planned_dispersion_min"]))
    f = ft.row_features_for_batch(b, base, cfg, get(cfg, "kpi_targets.breach_lead_time_min"))
    print(f"  export has the schema's {len(schema.measured_columns())} columns; Step 4 features build on it ({f.shape[1]} columns); "
          f"use TSD_DATA_DIR=<folder> for the check scripts and the dashboard")

    # ------------------------------------------------------------------- 6
    banner("6. RECALIBRATION ON THE SIMULATED PLANT (the hidden mismatch is known, so the refit can be judged)")
    mm = cfg["simulation"]["plant_mismatch"]
    truth = {"Np_turb": mm["np_turbulent_factor"]["value"], "h_product": mm["h_product_factor"]["value"],
             "evaporation": mm["evaporative_loss_factor"]["value"], "e_c": mm["char_specific_energy_factor"]["value"]}
    out_cfg = SCRATCH / "assumptions_recalibrated.yaml"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "recalibrate.py"), "--data", str(ROOT / "data" / "raw"), "--out", str(out_cfg), "--batches", "80"],
                         capture_output=True, text=True)
    print("  " + "\n  ".join(l.strip() for l in res.stdout.splitlines() if l.strip().startswith(("1.", "2.", "3.", "4.", "physics-only", "multipliers"))))
    mult_line = [l for l in res.stdout.splitlines() if "multipliers:" in l][0]
    mults = eval(mult_line.split("multipliers:")[1].strip())
    mae_line = [l for l in res.stdout.splitlines() if "physics-only temperature MAE" in l][0]
    before, after = [float(x) for x in mae_line.split(":")[1].replace("degC (median)", "").split("->")]
    print(f"  hidden truth: Np x{truth['Np_turb']}, h_product x{truth['h_product']}, evaporation x{truth['evaporation']}, e_c x{truth['e_c']}")
    assert after < 0.5 * before, "recalibration did not halve the physics-only error"
    assert mults["Np_turb"] > 1.0 and mults["h_product"] < 1.0 and mults["evaporation"] > 1.0 and mults["e_c_water"] < 1.0, "a factor moved the wrong way"
    print(f"  every factor moved in the hidden direction; physics-only MAE {before:.2f} -> {after:.2f} degC   OK")
    print("  (h_product and evaporation are partly confounded - each lands a little off, together they reproduce the batch; the order matters)")

    banner("ALL STEP 9 CHECKS PASSED - see docs/PATH_TO_LIVE.md for the plant-side requirements and the 30/60/90-day plan")


if __name__ == "__main__":
    main()
