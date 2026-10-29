"""
mock_plc_opcua.py - an OPC UA server that behaves like the plant's DCS/PLC + MES for the disperser.

It exposes every NodeId in config/tag_map.csv and replays simulated batches through them at
`--speed` x real time. Use it for the integration test (check_step9.py), for a factory
acceptance test of the twin before the plant exists, and to train operators on the dashboard.

    python scripts/mock_plc_opcua.py --port 4840 --speed 20 --batches TSD-0176,TSD-0177

Endpoint: opc.tcp://127.0.0.1:<port>/tsd/   (no security - this is a test server)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tsd_twin import live, schema  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

STEP_TO_MES = {"charge_liquids": "CHARGE", "pigment_addition": "PIGMENT", "high_speed_dispersion": "DISPERSE", "letdown": "LETDOWN"}


async def serve(port: int, speed: float, batch_ids: list[str] | None, loops: int, ready_file: str | None, rows: int | None = None,
                start_delay: float = 0.0) -> None:
    from asyncua import Server, ua
    cfg = load_config()
    period = float(get(cfg, "data.sample_period_s"))
    tmap = live.TagMap(ROOT / "config" / "tag_map.csv", ROOT / "config" / "step_map.csv")
    ts = pd.read_csv(ROOT / "data" / "raw" / "timeseries.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(ROOT / "data" / "raw" / "batches.csv").set_index("batch_id")
    if batch_ids:
        ts = ts[ts["batch_id"].isin(batch_ids)]

    server = Server()
    await server.init()
    server.set_endpoint(f"opc.tcp://127.0.0.1:{port}/tsd/")
    server.set_server_name("TSD mock PLC/MES")
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
    ns = await server.register_namespace("http://tsd-twin/mock")
    objects = server.nodes.objects
    folder = await objects.add_folder(ns, "TSD01")
    nodes = {}
    for col, spec in tmap.specs.items():
        string_tag = col in ("batch_id", "step", "recipe_id", "product_family")
        init = "" if string_tag else (-1.0 if col == "plc_heartbeat" else 0.0)   # heartbeat starts off the replay's first value
        # NodeId "ns=2;s=TSD01.X.Y" -> create with the exact string identifier so clients can address it
        ident = spec.address.rsplit("s=", 1)[1]            # "ns=2;s=TSD01.TI1201.PV" -> "TSD01.TI1201.PV"
        var = await folder.add_variable(ua.NodeId(ident, ns), ident.split(".")[-1] if "." in ident else ident, init)
        await var.set_writable()
        nodes[col] = var
    await server.start()
    print(f"mock PLC serving {len(nodes)} tags at opc.tcp://127.0.0.1:{port}/tsd/  (speed x{speed})", flush=True)
    if ready_file:
        Path(ready_file).write_text("ready")
    await asyncio.sleep(start_delay)
    try:
        for _ in range(loops):
            for bid, b in ts.groupby("batch_id", sort=False):
                m = meta.loc[bid]
                for col, val in [("batch_mass_nominal_kg", float(m["batch_mass_nominal_kg"])), ("cw_flow_setpoint_m3h", float(m["cw_flow_setpoint_m3h"])),
                                 ("planned_dispersion_min", float(m["planned_dispersion_min"]))]:
                    await nodes[col].write_value(val)
                for k, (_, r) in enumerate(b.iterrows()):
                    if rows and k >= rows:
                        break
                    for col in schema.measured_columns():
                        if col in ("timestamp", "t_s"):
                            continue
                        if col in ("hegman", "viscosity_ku"):
                            if pd.notna(r[col]):
                                await nodes[col].write_value(float(r[col]))
                            continue
                        v = STEP_TO_MES[r["step"]] if col == "step" else (str(r[col]) if col in ("batch_id", "recipe_id", "product_family") else float(r[col]))
                        await nodes[col].write_value(v)
                    await nodes["plc_heartbeat"].write_value(float(r["t_s"]))     # heartbeat LAST = "all values of this cycle are written"
                    await asyncio.sleep(period / speed if speed > 0 else 0)
                await nodes["batch_id"].write_value("")
                await nodes["step"].write_value("IDLE")
                await asyncio.sleep(period / speed if speed > 0 else 0)
        print("mock PLC: replay finished; staying up for 120 s", flush=True)
        await asyncio.sleep(120)
    finally:
        await server.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4840)
    ap.add_argument("--speed", type=float, default=20.0, help="replay speed multiplier (0 = as fast as possible)")
    ap.add_argument("--batches", type=str, default="", help="comma-separated batch ids (default: all)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--ready-file", type=str, default=None)
    ap.add_argument("--rows", type=int, default=None, help="replay only the first N rows of each batch (tests)")
    ap.add_argument("--start-delay", type=float, default=0.0, help="seconds to wait after start-up before the replay begins")
    a = ap.parse_args()
    asyncio.run(serve(a.port, a.speed, [x for x in a.batches.split(",") if x] or None, a.loops, a.ready_file, a.rows, a.start_delay))


if __name__ == "__main__":
    main()
