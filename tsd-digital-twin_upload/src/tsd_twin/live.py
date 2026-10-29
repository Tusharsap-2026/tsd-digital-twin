"""
live.py - the path from PLC/DCS tags to the twin (Step 9).

    tag stream  ->  TagBuffer  ->  BatchAssembler (10 s canonical rows)  ->  Historian (SQLite)
                                          |
                                          v
                                      LiveTwin: runtime.prepare_batch on the rows so far,
                                      publishes state + alerts, writes the batch verdict at the end

Sources (all push tag updates into the same TagBuffer, so everything downstream is identical):
    ReplaySource   - replays a canonical CSV (the simulator's or a historian export), accelerated
    OpcUaSource    - OPC UA client subscription (asyncua) using config/tag_map.csv NodeIds
    MqttSource     - MQTT subscriber (paho-mqtt) using config/tag_map.csv topics, JSON payloads

Nothing downstream of the TagBuffer knows which source is connected. That is what makes
the switch from simulation to the plant a configuration change.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from . import runtime as rt, schema
from .config import get

NAN = float("nan")


# ================================================================ tag map ==
@dataclass
class TagSpec:
    column: str
    source: str
    address: str
    topic: str
    scale: float
    offset: float
    deadband: float
    max_age_s: float
    required: bool
    description: str


class TagMap:
    """config/tag_map.csv: canonical column <-> plant tag, with scaling and staleness rules."""

    def __init__(self, path: Path | str, step_map_path: Path | str | None = None):
        df = pd.read_csv(path).fillna({"deadband": 0.0, "scale": 1.0, "offset": 0.0, "max_age_s": 60.0})
        self.specs: dict[str, TagSpec] = {}
        for _, r in df.iterrows():
            self.specs[r["canonical_column"]] = TagSpec(r["canonical_column"], r["source"], str(r["tag_address"]), str(r["mqtt_topic"]),
                                                       float(r["scale"]), float(r["offset"]), float(r["deadband"]), float(r["max_age_s"]),
                                                       str(r["required"]).lower() == "yes", str(r["description"]))
        self.by_address = {s.address: s.column for s in self.specs.values()}
        self.by_topic = {s.topic: s.column for s in self.specs.values()}
        self.step_map: dict[str, str] = {}
        if step_map_path and Path(step_map_path).exists():
            sm = pd.read_csv(step_map_path).fillna("")
            self.step_map = {str(r["mes_phase_name"]).upper(): str(r["canonical_step"]) for _, r in sm.iterrows()}

    def to_canonical_step(self, phase_name: str) -> str:
        """MES phase name -> canonical step ('' = not in a modelled phase)."""
        if phase_name in schema.STEP_NAMES:
            return phase_name
        return self.step_map.get(str(phase_name).upper(), "")

    def columns(self) -> list[str]:
        return list(self.specs)


# ============================================================= tag buffer ==
@dataclass
class TagValue:
    value: object
    ts: datetime
    quality: str = "good"          # good | stale | bad


class TagBuffer:
    """Latest value of every canonical tag, thread-safe. Sources write, the assembler reads."""

    def __init__(self, tag_map: TagMap):
        self.tag_map = tag_map
        self._lock = threading.Lock()
        self._values: dict[str, TagValue] = {}
        self.n_updates = 0
        self.clock_override: datetime | None = None     # accelerated tests: stamp updates with the assembler clock

    def update(self, column: str, value, ts: datetime | None = None, quality: str = "good") -> None:
        spec = self.tag_map.specs.get(column)
        if spec is None:
            return
        if self.clock_override is not None and column != "plc_heartbeat":
            ts = self.clock_override
        elif ts is not None and ts.tzinfo is not None:
            ts = ts.astimezone().replace(tzinfo=None)      # OPC UA source time stamps are UTC-aware; keep everything naive local
        if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
            value = float(value) * spec.scale + spec.offset
        with self._lock:
            self._values[column] = TagValue(value, ts or datetime.now(), quality)
            self.n_updates += 1

    def update_by_address(self, address: str, value, ts=None, quality="good") -> None:
        col = self.tag_map.by_address.get(address)
        if col:
            self.update(col, value, ts, quality)

    def update_by_topic(self, topic: str, value, ts=None, quality="good") -> None:
        col = self.tag_map.by_topic.get(topic)
        if col:
            self.update(col, value, ts, quality)

    def snapshot(self, now: datetime) -> dict[str, TagValue]:
        """Copy of the buffer with staleness applied against `now`. Values are latched (a tag that
        has not changed is still good); liveness comes from tags with max_age_s > 0 (the PLC
        heartbeat): if the heartbeat is stale, every PLC-sourced value is marked stale."""
        with self._lock:
            out = {}
            plc_dead = False
            wall = datetime.now()
            for col, tv in self._values.items():
                spec = self.tag_map.specs[col]
                ref = wall if col == "plc_heartbeat" else now
                stale = spec.max_age_s > 0 and (ref - tv.ts).total_seconds() > spec.max_age_s
                if stale and col == "plc_heartbeat":
                    plc_dead = True
                out[col] = TagValue(tv.value, tv.ts, "stale" if stale and tv.quality == "good" else tv.quality)
            if "plc_heartbeat" in self.tag_map.specs and "plc_heartbeat" not in self._values:
                plc_dead = True                           # never seen a heartbeat: PLC link not up
            if plc_dead:
                for col, tv in out.items():
                    if self.tag_map.specs[col].source == "PLC" and tv.quality == "good":
                        out[col] = TagValue(tv.value, tv.ts, "stale")
            return out


# ========================================================= batch assembler ==
class BatchAssembler:
    """Turns the tag buffer into canonical 10 s rows and groups them into batches.
    A batch starts when batch_id becomes non-empty and the step is a modelled phase; it ends
    when batch_id changes or clears. Lab values are sparse: they are written once, at the row
    whose time stamp follows the sample's time stamp, then cleared."""

    def __init__(self, tag_map: TagMap, cfg: dict, on_row: Callable[[pd.DataFrame], None] | None = None,
                 on_batch_end: Callable[[pd.DataFrame], None] | None = None):
        self.tag_map, self.cfg = tag_map, cfg
        self.period = float(get(cfg, "data.sample_period_s"))
        self.on_row, self.on_batch_end = on_row, on_batch_end
        self.rows: list[dict] = []
        self.batch_id: str | None = None
        self.t0: datetime | None = None
        self.pending_lab: dict[str, float] = {}
        self.quality_log: list[dict] = []

    def lab_result(self, column: str, value: float) -> None:
        """A lab result arriving out of band (LIMS message, operator entry)."""
        self.pending_lab[column] = float(value)

    def tick(self, now: datetime, buf: TagBuffer) -> pd.DataFrame | None:
        snap = buf.snapshot(now)
        bid = str(snap["batch_id"].value) if "batch_id" in snap and snap["batch_id"].value not in (None, "", "nan") else None
        step = self.tag_map.to_canonical_step(str(snap["step"].value)) if "step" in snap else ""
        if bid != self.batch_id:                               # batch boundary
            if self.batch_id is not None:
                self._finish()
            self.batch_id, self.t0, self.rows = bid, now, []
        if bid is None or not step:
            return None
        row = {"batch_id": bid, "timestamp": now, "t_s": round((now - self.t0).total_seconds() / self.period) * self.period, "step": step}
        bad = []
        for col in schema.measured_columns():
            if col in row or col in ("hegman", "viscosity_ku"):
                continue
            tv = snap.get(col)
            if tv is None or tv.quality != "good":
                row[col] = NAN
                if tv is None or self.tag_map.specs.get(col, TagSpec(col, "", "", "", 1, 0, 0, 0, False, "")).required:
                    bad.append(col)
            else:
                row[col] = tv.value
        for col in ("hegman", "viscosity_ku"):
            row[col] = self.pending_lab.pop(col, NAN)
        if bad:
            self.quality_log.append({"batch_id": bid, "timestamp": now, "missing_or_stale": ",".join(bad)})
        self.rows.append(row)
        df = pd.DataFrame(self.rows)
        if self.on_row:
            self.on_row(df)
        return df

    def _finish(self) -> None:
        if self.rows and self.on_batch_end:
            self.on_batch_end(pd.DataFrame(self.rows))

    def close(self) -> None:
        self._finish()
        self.batch_id, self.rows = None, []


# ========================================================== tick scheduler ==
class HeartbeatTicker:
    """Decides WHEN the assembler builds a row. Preferred: on the PLC heartbeat (a free-running
    seconds counter), one row per sample period of PLC time, so the twin cycles on the plant's
    clock and the test rig can run accelerated. Fallback (no heartbeat tag): every tick_s of
    wall-clock time."""

    def __init__(self, buf: TagBuffer, period_s: float, tick_s: float | None = None, settle_s: float = 0.2):
        self.buf, self.period, self.tick_s, self.settle_s = buf, period_s, tick_s, settle_s
        self.hb0: float | None = None
        self.hb_last: float | None = None
        self.ticks = 0
        self.pending: list[datetime] = []          # rows detected on the previous poll, built on this one (settle time)
        self.clock0 = datetime.now()
        self.wall_last = datetime.now()
        self.uses_heartbeat = "plc_heartbeat" in buf.tag_map.specs

    def poll_interval(self) -> float:
        """Heartbeat mode polls every settle_s (also the settle time after a heartbeat change, so it
        must be a small fraction of the PLC cycle); wall-clock mode sleeps the tick."""
        return self.settle_s if self.uses_heartbeat else (self.tick_s or self.period)

    def due(self) -> list[datetime]:
        """Clocks of the rows that are due now (usually zero or one; more after a hiccup)."""
        if not self.uses_heartbeat:
            now = datetime.now()
            if (now - self.wall_last).total_seconds() >= (self.tick_s or self.period) - 1e-6:
                self.wall_last = now
                return [now]
            return []
        ready, self.pending = self.pending, []      # one poll interval of settle time after the heartbeat changed
        with self.buf._lock:
            tv = self.buf._values.get("plc_heartbeat")
        if tv is None:
            return ready
        hb = float(tv.value)
        if self.hb0 is None:                       # first value seen: one row (the assembler ignores it if no batch is active)
            self.hb0 = self.hb_last = hb
            self.pending = [self.clock0]
            return ready
        if hb == self.hb_last:
            return ready
        # one row per PLC cycle; a jump of several periods (missed notifications) yields several rows.
        # Row clocks come from a tick counter so they sit exactly on the sample grid.
        n = max(1, min(10, int(round((hb - self.hb_last) / self.period))))
        out = []
        for _ in range(n):
            self.ticks += 1
            out.append(self.clock0 + timedelta(seconds=self.ticks * self.period))
        self.hb_last = hb
        self.pending = out
        return ready


# ================================================================ sources ==
class ReplaySource:
    """Replays a canonical CSV through the tag buffer at `speed` x real time, batch by batch.
    Lab values become out-of-band lab results, as they would from a LIMS."""

    def __init__(self, csv_path: Path | str, buf: TagBuffer, assembler: BatchAssembler, batch_ids: list[str] | None = None,
                 speed: float = 50.0, batch_meta: pd.DataFrame | None = None):
        self.ts = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if batch_ids:
            self.ts = self.ts[self.ts["batch_id"].isin(batch_ids)]
        self.buf, self.asm, self.speed, self.meta = buf, assembler, speed, batch_meta

    def run(self, max_batches: int | None = None, realtime_clock: bool = False) -> None:
        period = self.asm.period
        n = 0
        clock = datetime.now()
        for bid, b in self.ts.groupby("batch_id", sort=False):
            if max_batches is not None and n >= max_batches:
                break
            n += 1
            if self.meta is not None and bid in self.meta.index:
                m = self.meta.loc[bid]
                self.buf.update("batch_mass_nominal_kg", float(m["batch_mass_nominal_kg"]), clock)
                self.buf.update("cw_flow_setpoint_m3h", float(m["cw_flow_setpoint_m3h"]), clock)
                self.buf.update("planned_dispersion_min", float(m["planned_dispersion_min"]), clock)
            for _, r in b.iterrows():
                clock = r["timestamp"] if realtime_clock else clock + timedelta(seconds=period)
                self.buf.update("plc_heartbeat", float(r["t_s"]), clock)
                for col in schema.measured_columns():
                    if col in ("timestamp", "t_s", "hegman", "viscosity_ku"):
                        continue
                    self.buf.update(col, r[col], clock)
                for col in ("hegman", "viscosity_ku"):
                    if pd.notna(r[col]):
                        self.asm.lab_result(col, float(r[col]))
                self.asm.tick(clock, self.buf)
                if self.speed > 0:
                    time.sleep(period / self.speed)
            # end of batch: MES clears the batch id
            clock = clock + timedelta(seconds=period)
            self.buf.update("batch_id", "", clock)
            self.buf.update("step", "IDLE", clock)
            self.asm.tick(clock, self.buf)
        self.asm.close()


class OpcUaSource:
    """OPC UA client: subscribes to every NodeId in the tag map and pushes data changes into the
    buffer. Run with `await source.run(stop_event)` inside an asyncio loop; the assembler is
    ticked every sample period from the same loop."""

    def __init__(self, endpoint: str, buf: TagBuffer, assembler: BatchAssembler, username: str | None = None,
                 password: str | None = None, publishing_ms: int = 500, tick_s: float | None = None, settle_s: float = 0.2):
        self.endpoint, self.buf, self.asm = endpoint, buf, assembler
        self.username, self.password, self.publishing_ms = username, password, publishing_ms
        self.tick_s = tick_s                      # wall-clock ticking when there is no heartbeat tag
        self.settle_s = settle_s                  # settle time after a heartbeat change before the row is built
        self.connected = False

    class _Handler:
        def __init__(self, outer, node_to_col):
            self.outer, self.node_to_col = outer, node_to_col

        def datachange_notification(self, node, val, data):
            col = self.node_to_col.get(node.nodeid.to_string())
            if col is None:
                return
            if self.outer.buf.tag_map.specs[col].source == "LAB":          # lab results are events, not states
                try:
                    self.outer.asm.lab_result(col, float(val))
                except (TypeError, ValueError):
                    pass
                return
            ts = getattr(getattr(data, "monitored_item", None), "Value", None)
            src_ts = getattr(ts, "SourceTimestamp", None) if ts is not None else None
            q = "good"
            sc = getattr(ts, "StatusCode", None) if ts is not None else None
            if sc is not None and not sc.is_good():
                q = "bad"
            self.outer.buf.update(col, val, src_ts or datetime.now(), q)

    async def run(self, stop_event, tick_cb: Callable[[datetime], None] | None = None) -> None:
        import asyncio
        from asyncua import Client, ua
        client = Client(url=self.endpoint)
        if self.username:
            client.set_user(self.username)
            client.set_password(self.password or "")
        async with client:
            self.connected = True
            nodes, node_to_col = [], {}
            for col, spec in self.buf.tag_map.specs.items():
                try:
                    node = client.get_node(spec.address)
                    nodes.append(node)
                    node_to_col[node.nodeid.to_string()] = col
                except Exception:          # tag not on this server: leave NaN, the assembler logs it
                    continue
            sub = await client.create_subscription(self.publishing_ms, self._Handler(self, node_to_col))
            await sub.subscribe_data_change(nodes)
            ticker = HeartbeatTicker(self.buf, self.asm.period, self.tick_s, self.settle_s)
            while not stop_event.is_set():
                await asyncio.sleep(ticker.poll_interval())
                for clock in ticker.due():
                    self.buf.clock_override = clock if ticker.uses_heartbeat else None
                    # the twin run inside tick() can take ~0.1 s: keep it off the subscription loop
                    await asyncio.get_running_loop().run_in_executor(None, self.asm.tick, clock, self.buf)
                    if tick_cb:
                        tick_cb(clock)
            await sub.delete()
        self.connected = False


class MqttSource:
    """MQTT subscriber: one topic per tag (config/tag_map.csv), payload either a bare number or
    JSON {"value": 41.2, "ts": "2027-08-01T10:15:30", "quality": "good"}. Lab topics carry results
    out of band. The assembler is ticked by a timer thread."""

    def __init__(self, host: str, buf: TagBuffer, assembler: BatchAssembler, port: int = 1883, username: str | None = None,
                 password: str | None = None, client_id: str = "tsd-twin", tick_s: float | None = None, settle_s: float = 0.2):
        self.host, self.port, self.buf, self.asm = host, port, buf, assembler
        self.username, self.password, self.client_id = username, password, client_id
        self.tick_s, self.settle_s = tick_s, settle_s
        self._stop = threading.Event()

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                d = {"value": float(payload)}
            if not isinstance(d, dict):
                d = {"value": d}
            ts = datetime.fromisoformat(d["ts"]) if "ts" in d else datetime.now()
            col = self.buf.tag_map.by_topic.get(msg.topic)
            if col in ("hegman", "viscosity_ku"):
                self.asm.lab_result(col, float(d["value"]))
            elif col:
                self.buf.update(col, d["value"], ts, d.get("quality", "good"))
        except Exception:
            pass

    def run(self, tick_cb: Callable[[datetime], None] | None = None) -> None:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        if self.username:
            client.username_pw_set(self.username, self.password)
        client.on_message = self._on_message
        client.connect(self.host, self.port, keepalive=60)
        for spec in self.buf.tag_map.specs.values():
            client.subscribe(spec.topic)
        client.loop_start()
        try:
            ticker = HeartbeatTicker(self.buf, self.asm.period, self.tick_s, self.settle_s)
            while not self._stop.is_set():
                time.sleep(ticker.poll_interval())
                for clock in ticker.due():
                    self.buf.clock_override = clock if ticker.uses_heartbeat else None
                    self.asm.tick(clock, self.buf)
                    if tick_cb:
                        tick_cb(clock)
        finally:
            client.loop_stop()
            client.disconnect()

    def stop(self) -> None:
        self._stop.set()


# ============================================================== historian ==
class Historian:
    """SQLite store: canonical rows, twin state, batch verdicts, alerts, quality log.
    One file, no server - enough for one disperser and years of 10 s data (~3 MB per 100 batches).
    Swap for the plant historian / TimescaleDB by re-implementing these four methods."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS timeseries (batch_id TEXT, timestamp TEXT, t_s REAL, step TEXT, data TEXT, PRIMARY KEY (batch_id, t_s))")
        self.con.execute("CREATE TABLE IF NOT EXISTS twin_state (batch_id TEXT, timestamp TEXT, t_s REAL, T_meas REAL, T_nowcast REAL, T_next5 REAL, "
                         "breach_eta_min REAL, alert_code TEXT, likely_cause TEXT, energy_cum_kwh REAL, kw_resid_pct REAL, u_drop_pct REAL, PRIMARY KEY (batch_id, t_s))")
        self.con.execute("CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, finished TEXT, verdict TEXT)")
        self.con.execute("CREATE TABLE IF NOT EXISTS alerts (batch_id TEXT, alert TEXT, raised_at_min REAL, detail TEXT)")
        self.con.execute("CREATE TABLE IF NOT EXISTS quality_log (batch_id TEXT, timestamp TEXT, missing_or_stale TEXT)")
        self.con.commit()

    def write_row(self, row: pd.Series) -> None:
        data = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.isoformat() if isinstance(v, (datetime, pd.Timestamp)) else v))
                for k, v in row.items() if k not in ("batch_id", "timestamp", "t_s", "step")}
        self.con.execute("INSERT OR REPLACE INTO timeseries VALUES (?,?,?,?,?)",
                         (row["batch_id"], pd.Timestamp(row["timestamp"]).isoformat(), float(row["t_s"]), row["step"], json.dumps(data)))

    def write_state(self, s: pd.Series) -> None:
        self.con.execute("INSERT OR REPLACE INTO twin_state VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (s["batch_id"], pd.Timestamp(s["timestamp"]).isoformat(), float(s["t_s"]), float(s["T_meas"]), float(s["T_nowcast"]),
                          None if pd.isna(s["T_next5"]) else float(s["T_next5"]), None if pd.isna(s["breach_eta_min"]) else float(s["breach_eta_min"]),
                          s["alert_code"], s["likely_cause"], float(s["energy_cum_kwh"]),
                          None if pd.isna(s["kw_resid_pct"]) else float(s["kw_resid_pct"]), None if pd.isna(s["u_drop_pct"]) else float(s["u_drop_pct"])))

    def write_batch(self, verdict: dict, alerts: pd.DataFrame) -> None:
        self.con.execute("INSERT OR REPLACE INTO batches VALUES (?,?,?)", (verdict["batch_id"], datetime.now().isoformat(), json.dumps(verdict, default=str)))
        for _, a in alerts.iterrows():
            self.con.execute("INSERT INTO alerts VALUES (?,?,?,?)", (a["batch_id"], a["alert"], float(a["raised_at_min"]), json.dumps(a.to_dict(), default=str)))

    def write_quality(self, entries: list[dict]) -> None:
        for e in entries:
            self.con.execute("INSERT INTO quality_log VALUES (?,?,?)", (e["batch_id"], pd.Timestamp(e["timestamp"]).isoformat(), e["missing_or_stale"]))

    def commit(self) -> None:
        self.con.commit()

    def read_batch(self, batch_id: str) -> pd.DataFrame:
        rows = self.con.execute("SELECT batch_id, timestamp, t_s, step, data FROM timeseries WHERE batch_id=? ORDER BY t_s", (batch_id,)).fetchall()
        recs = []
        for bid, ts, t_s, step, data in rows:
            d = json.loads(data)
            d.update({"batch_id": bid, "timestamp": pd.Timestamp(ts), "t_s": t_s, "step": step})
            recs.append(d)
        return pd.DataFrame(recs)

    def counts(self) -> dict:
        return {t: self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("timeseries", "twin_state", "batches", "alerts", "quality_log")}


# ================================================================ live twin ==
class LiveTwin:
    """Runs the Step 7 runtime incrementally on the rows assembled so far.
    every_n_rows: how often to re-run (3 = every 30 s; a full-batch run costs ~0.1 s)."""

    def __init__(self, cfg: dict, tm, baselines: rt.Baselines, thr: rt.Thresholds, historian: Historian | None = None,
                 every_n_rows: int = 3, on_state: Callable[[pd.Series], None] | None = None,
                 on_alert: Callable[[str, pd.Series], None] | None = None, min_rows: int = 12):
        self.cfg, self.tm, self.bl, self.thr = cfg, tm, baselines, thr
        self.hist, self.every, self.min_rows = historian, every_n_rows, min_rows
        self.on_state, self.on_alert = on_state, on_alert
        self.last_state: pd.Series | None = None
        self.active_alerts: set[str] = set()
        self.n_runs = 0
        self.run_times: list[float] = []

    def _meta(self, df: pd.DataFrame, buf: TagBuffer) -> tuple[float, float]:
        snap = buf.snapshot(datetime.now())
        nominal = float(snap["batch_mass_nominal_kg"].value) if "batch_mass_nominal_kg" in snap else float(df["batch_mass_kg"].max())
        flow_sp = float(snap["cw_flow_setpoint_m3h"].value) if "cw_flow_setpoint_m3h" in snap else float(df["cw_flow_m3h"].median())
        return nominal, flow_sp

    def on_row(self, df: pd.DataFrame, buf: TagBuffer) -> None:
        if self.hist:
            self.hist.write_row(df.iloc[-1])
        if len(df) < self.min_rows or len(df) % self.every:
            return
        nominal, flow_sp = self._meta(df, buf)
        t = time.time()
        run = rt.prepare_batch(self._clean(df), self.cfg, self.tm, nominal, flow_sp, self.thr, self.bl)
        self.run_times.append(time.time() - t)
        self.n_runs += 1
        state = run.iloc[-1]
        self.last_state = state
        if self.hist:
            self.hist.write_state(state)
            self.hist.commit()
        if self.on_state:
            self.on_state(state)
        now_active = {c for c in ["TEMP_LIMIT", "TEMP_PREALARM", "LOW_CW_FLOW", "OVERCHARGE", "JACKET_FOULING", "BLADE_WEAR"] if run[f"rule_{c}"].iloc[-1]}
        for c in now_active - self.active_alerts:
            if self.on_alert:
                self.on_alert(c, state)
        self.active_alerts = now_active

    def on_batch_end(self, df: pd.DataFrame, buf: TagBuffer) -> dict | None:
        if len(df) < self.min_rows:
            return None
        nominal, flow_sp = self._meta(df, buf)
        run = rt.prepare_batch(self._clean(df), self.cfg, self.tm, nominal, flow_sp, self.thr, self.bl)
        verdict = rt.batch_verdict(run)
        snap = buf.snapshot(datetime.now())
        verdict.update({"nominal_mass_kg": nominal, "cw_flow_setpoint_m3h": flow_sp,
                        "planned_dispersion_min": float(snap["planned_dispersion_min"].value) if "planned_dispersion_min" in snap else None})
        alerts = rt.alert_table(run)
        if self.hist:
            self.hist.write_batch(verdict, alerts)
            self.hist.commit()
        self.active_alerts = set()
        return verdict

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill short gaps in the dense tags (stale samples); lab columns stay sparse."""
        out = df.copy()
        dense = [c for c in schema.measured_columns() if c not in ("hegman", "viscosity_ku", "batch_id", "timestamp", "t_s", "step")]
        out[dense] = out[dense].ffill(limit=6).bfill(limit=6)
        return out
