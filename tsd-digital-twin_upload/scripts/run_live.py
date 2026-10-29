"""
run_live.py - run the twin against a live source and keep the historian up to date.

    python scripts/run_live.py --source replay --speed 50 --batches TSD-0176,TSD-0177
    python scripts/run_live.py --source opcua --endpoint opc.tcp://10.20.30.40:4840/dcs/ --user twin --password ***
    python scripts/run_live.py --source mqtt --host 10.20.30.41 --port 1883

Writes data/live/historian.sqlite (rows, twin state, batch verdicts, alerts, tag-quality log)
and prints every alert as it is raised. The dashboard can read the same historian
(see docs/PATH_TO_LIVE.md, "dashboard on live data").
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tsd_twin import live, models as md, runtime as rt  # noqa: E402
from tsd_twin.config import load_config  # noqa: E402


def build(args) -> tuple[live.TagBuffer, live.BatchAssembler, live.LiveTwin, live.Historian]:
    cfg = load_config()
    tm, _, _ = md.load_models(ROOT / "models")
    bl = rt.Baselines.load(ROOT / "models" / "runtime_calibration.json")
    thr = rt.Thresholds.from_config(cfg)
    tmap = live.TagMap(ROOT / "config" / "tag_map.csv", ROOT / "config" / "step_map.csv")
    buf = live.TagBuffer(tmap)
    hist = live.Historian(args.historian)

    def on_alert(code, s):
        cause = f" - {s['likely_cause']}" if s["likely_cause"] else ""
        eta = "" if pd.isna(s["breach_eta_min"]) else f" (breach ETA {s['breach_eta_min']:.0f} min)"
        print(f"  ALERT {s['batch_id']} t={s['t_min']:.1f} min  {code}{eta}{cause}", flush=True)

    def on_state(s):
        if int(round(s["t_s"])) % 300 == 0:          # a line every 5 min of batch time
            print(f"  {s['batch_id']} t={s['t_min']:5.1f} {s['step']:22s} T={s['T_meas']:5.1f} nowcast={s['T_nowcast']:5.1f} "
                  f"+5min={'  -  ' if pd.isna(s['T_next5']) else f'{s['T_next5']:5.1f}'} kW={s['hsd_kw']:5.1f} E={s['energy_cum_kwh']:5.1f} kWh  {s['alert_code']}", flush=True)

    twin = live.LiveTwin(cfg, tm, bl, thr, hist, every_n_rows=args.every, on_state=on_state, on_alert=on_alert)
    asm = live.BatchAssembler(tmap, cfg, on_row=lambda df: twin.on_row(df, buf),
                              on_batch_end=lambda df: _end(twin, df, buf, hist, asm))
    return buf, asm, twin, hist


def _end(twin, df, buf, hist, asm):
    v = twin.on_batch_end(df, buf)
    if v:
        print(f"  BATCH END {v['batch_id']}: {len(df)} rows, {v['energy_kwh']:.1f} kWh ({v['kwh_per_t']:.2f} kWh/t), peak {v['peak_T']:.1f} degC, "
              f"flags {v['flags']}, temperature {v['temp_flag']}, now-cast MAE {v['nowcast_mae']:.2f} degC", flush=True)
    if asm.quality_log:
        hist.write_quality(asm.quality_log)
        asm.quality_log.clear()
        hist.commit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["replay", "opcua", "mqtt"], default="replay")
    ap.add_argument("--speed", type=float, default=50.0, help="replay: x real time (0 = as fast as possible)")
    ap.add_argument("--batches", type=str, default="", help="replay: comma-separated batch ids")
    ap.add_argument("--csv", type=str, default=str(ROOT / "data" / "raw" / "timeseries.csv"))
    ap.add_argument("--endpoint", type=str, default="opc.tcp://127.0.0.1:4840/tsd/")
    ap.add_argument("--user", type=str, default=None)
    ap.add_argument("--password", type=str, default=None)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--historian", type=str, default=str(ROOT / "data" / "live" / "historian.sqlite"))
    ap.add_argument("--every", type=int, default=3, help="re-run the twin every N rows (3 = 30 s)")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None, help="opcua/mqtt: stop after this many seconds")
    ap.add_argument("--tick", type=float, default=None, help="opcua/mqtt: wall-clock tick when there is no PLC heartbeat tag (normally unused)")
    ap.add_argument("--publishing-ms", type=int, default=500, help="opcua: subscription publishing interval")
    ap.add_argument("--settle", type=float, default=0.2, help="seconds after a heartbeat change before the row is built (fraction of the PLC cycle)")
    args = ap.parse_args()
    buf, asm, twin, hist = build(args)
    print(f"twin live - source {args.source}, historian {args.historian}", flush=True)
    if args.source == "replay":
        meta = pd.read_csv(ROOT / "data" / "raw" / "batches.csv").set_index("batch_id")
        src = live.ReplaySource(args.csv, buf, asm, [x for x in args.batches.split(",") if x] or None, args.speed, meta)
        src.run(max_batches=args.max_batches)
    elif args.source == "opcua":
        stop = asyncio.Event()
        src = live.OpcUaSource(args.endpoint, buf, asm, args.user, args.password, publishing_ms=args.publishing_ms, tick_s=args.tick, settle_s=args.settle)

        async def runner():
            if args.timeout:
                asyncio.get_running_loop().call_later(args.timeout, stop.set)
            await src.run(stop)
            asm.close()
        asyncio.run(runner())
    else:
        src = live.MqttSource(args.host, buf, asm, args.port, args.user, args.password, tick_s=args.tick, settle_s=args.settle)
        if args.timeout:
            threading.Timer(args.timeout, src.stop).start()
        src.run()
        asm.close()
    print(f"historian: {hist.counts()}  | twin runs {twin.n_runs}, mean {1000 * sum(twin.run_times) / max(len(twin.run_times), 1):.0f} ms", flush=True)


if __name__ == "__main__":
    main()
