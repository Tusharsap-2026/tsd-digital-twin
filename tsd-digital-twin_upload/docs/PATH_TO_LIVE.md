# Path to live data — connecting the TSD twin to the plant

This document covers what has to be true on the plant side, how the twin connects, what
happens on the first real batches, and the 30/60/90-day plan. Everything described here is
implemented in `src/tsd_twin/live.py`, `config/tag_map.csv`, `scripts/run_live.py`,
`scripts/mock_plc_opcua.py`, `scripts/export_historian.py` and `scripts/recalibrate.py`, and
tested end to end against a mock OPC UA server in `scripts/check_step9.py`.

## 1. Architecture

```
 Level 1/2   PLC / DCS (disperser skid)        MES (batch, phase, recipe)       LIMS / operator entry
                |  OPC UA server (preferred)       |  OPC UA tags or MQTT             |  Hegman, viscosity
                |  or MQTT via edge gateway         |                                   |
 Level 3     ---+-----------------------------------+-----------------------------------+--- plant network (OT)
                |
             TagBuffer  <- one canonical column per plant tag (config/tag_map.csv), latched values,
                |          liveness from the PLC heartbeat
             BatchAssembler -> one canonical row every 10 s of PLC time, batches cut on the MES batch id
                |
             Historian (SQLite, one file; swap for the site historian / TimescaleDB by re-implementing 4 methods)
                |
             LiveTwin -> runtime.prepare_batch on the rows so far every 30 s: now-cast, 5-min-ahead,
                |        breach ETA, fault rules, likely cause -> twin_state + alerts tables
             Dashboard (Streamlit) reads the same files / historian; alerts can also be published back
             to the DCS as OPC UA tags or to MQTT for the andon.
```

The twin runs on an edge PC or VM on the OT network (any Windows/Linux machine with Python 3.10;
one 10-s cycle costs ~0.1 s of CPU). It is **advisory**: it never writes a set-point. If the
plant later wants closed-loop action (e.g. trimming the HSD speed on a breach ETA), that goes
through the DCS with its own interlocks, not through this code.

## 2. OPC UA or MQTT?

| | OPC UA | MQTT (edge gateway, JSON or Sparkplug B) |
|---|---|---|
| Where it fits | DCS/PLC vendors ship an OPC UA server; MES often too | Site already has a broker / IIoT platform; many data consumers |
| Data model | Typed nodes, time stamps + status codes per value | Topics; you define the payload (we accept bare numbers or `{"value","ts","quality"}`) |
| Security | Certificates + user auth built in | TLS + broker ACLs |
| Reporting | Subscription, report on change, server time stamps | Publish on change / cyclic from the gateway |
| Our source class | `OpcUaSource` (asyncua) | `MqttSource` (paho-mqtt) |

**Recommendation for a greenfield DCS: OPC UA straight from the DCS server**, with the MES phase
and recipe tags exposed on the same server (most DCS/MES combinations allow this) or on the MES's
own OPC UA server. Use MQTT only if the site standardises on a broker for all analytics; then put
the mapping in the gateway and keep the same topic names as in the tag map.

Two things that matter with either protocol:

1. **Report-on-change means an unchanged value is never re-sent.** The twin therefore *latches*
   every value and takes liveness from a **PLC heartbeat** (a free-running seconds counter): if
   the heartbeat stops for 30 s, every PLC value is treated as bad. Do not use per-tag staleness
   timeouts; a steady anchor speed would look "stale".
2. **The assembler cycles on the PLC's clock, not the PC's.** A row is built one settle time
   (0.2 s) after each heartbeat change, so the row holds exactly the values of that PLC cycle.
   The PLC should write all values and then increment the heartbeat ("data ready").

## 3. Tag mapping table — `config/tag_map.csv`

One row per canonical column: source (PLC / MES / LAB), OPC UA NodeId, MQTT topic, units,
scale and offset (the twin expects SI; e.g. if the VFD reports power in hp, scale = 0.7457),
deadband (what the subscription may suppress), required flag and description. The NodeIds in
the file are placeholders in ISA style (`TI1201.PV`); replace them from the DCS I/O list. MES
phase names map to the four canonical steps through `config/step_map.csv` (edit the aliases to
your MES's names; phases not in the map, such as CIP, are ignored).

Instrument requirements the twin relies on (specify before the vendor freezes the skid):

- **Product temperature** TI1201: Pt100 class A in a thermowell reaching the millbase even at
  the smallest batch; 10-s sampling; this is the primary tag.
- **Jacket inlet/outlet** TI1202/1203: **matched pair**, class A or better, 0.1 °C resolution on
  the OPC tag — the fouling rule lives on their difference (1–3 K).
- **Chilled-water flow** FI1201: electromagnetic, ±1 %, and its set-point FIC1201.SP as a tag.
- **HSD power** JI1201: the VFD's *electrical power* parameter (not current × voltage estimated
  in the PLC), 0.1 kW resolution. **Anchor power** JI1202: 0.05 kW resolution — the anchor is
  the viscosity-lot gauge.
- **Speeds** SI1201/1202: VFD feedback; **set-point** SIC1201.SP as a tag (the twin needs to know
  what the operator asked for, to detect sag at the drive limit).
- **Load cells** WIT1201: 1 kg resolution, tared per batch.
- **Hall temperature** TI1290 near the vessel (or a daily value).
- **Heartbeat** PLC.Heartbeat: a seconds counter incremented after the cycle's values are written.
- **MES**: batch id, phase name, recipe id, product family, nominal batch size, solids, PVC,
  target Hegman, planned HSD time. **LAB**: Hegman and viscosity with the sampling time stamp
  (LIMS tag or an operator-entry screen on the HMI that writes a tag).

## 4. Sampling, deadbands, historian

- **Sampling: 10 s** for every PLC tag (the thermal time constant of the vessel is ~25 min; the
  fastest thing the twin tracks is a 5-min-ahead temperature; 10 s is already 30× faster than
  needed and keeps the historian small). Subscription publishing interval 500 ms.
- **Deadbands** small enough not to hide the signals the rules use: 0.05 °C on temperatures,
  0.1 kW on HSD power, 0.05 kW on anchor power, 0.1 m³/h on flow, 1 rpm, 2 kg.
- **Historian**: `live.Historian` writes five tables to one SQLite file (rows as JSON per
  sample, twin state, batch verdicts, alert episodes, tag-quality log). Volume: ~550 rows per
  batch, ~0.3 MB per batch, ~100 MB per year. The site historian (OSIsoft PI, Aveva, Ignition,
  TimescaleDB) can replace it by re-implementing `write_row / write_state / write_batch /
  read_batch`; the canonical row stays the contract. `scripts/export_historian.py` turns the
  historian into the canonical `timeseries.csv + batches.csv`, so the whole pipeline (features,
  models, what-if, alerts, dashboard) runs on real batches with `TSD_DATA_DIR=data/live`.

## 5. Linking to MES batch records

The twin's batch is **the MES's batch**: rows are cut on the MES batch-id tag, phases come from
the MES phase tag, and recipe attributes are read at batch start. That makes every twin result
joinable to the MES batch record by batch id (energy, kWh/t, peak temperature, time to target,
extensions, alerts). Three practical points:

- If the MES phase names differ from the four canonical steps (e.g. a separate "scrape-down"
  phase), map them in `step_map.csv`; a phase mapped to '' is ignored, one mapped to an existing
  step is merged into it.
- The **planned HSD time** and the lab checks are MES decisions: the energy *forecast* needs the
  planned time, and the lab result needs to arrive with the time it was *sampled*, not entered.
- Write the twin's batch verdict back to the MES batch record (or SAP PP-PI process messages)
  as a few numeric fields: kWh, kWh/t, peak °C, extensions, alert codes. The historian's
  `batches` table already holds them as JSON.

## 6. What to recalibrate first — and in what order

The simulated plant deliberately differs from the twin's physics (`simulation.plant_mismatch`),
and `scripts/recalibrate.py` was proven on it: with 80 batches it recovers the direction and
rough size of every hidden factor and cuts the physics-only temperature error from 0.86 to
0.19 °C. The order matters because each fit changes what the next one sees:

1. **HSD power number** (`physics.hsd_power.np_turbulent`) from measured vs physics shaft power
   at the actual speed. Needs: 20+ batches with the VFD power tag. Everything energetic depends
   on it.
2. **Product-side film** (`physics.heat_transfer.h_product_ref_w_m2k`) from the implied U of the
   measured jacket duty, taking the clean-jacket quartile. Needs the matched RTD pair.
3. **Evaporative loss** (`physics.losses.evaporative_loss_w_per_k.*`) by bisection until the
   temperature residual against the measured-power physics is zero. It is confounded with (2) —
   the fit lands where the two together reproduce the batch, which is what prediction needs.
4. **Grind energy** (`physics.grind.char_specific_energy_kwh_t.*`) from lab time-to-target vs
   physics, per family. Needs lab Hegman with sampling times.
5. Then re-run the pipeline (Steps 4–8: residual models, what-if calibration, runtime normal
   levels, alert thresholds) — about 10 minutes, all scripted.

Not identifiable from normal batches: the **tip-speed effectiveness curve** (which decides the
optimiser's "faster is cheaper" answer) and **phi_max**. Those need deliberate trials: the same
recipe at three HSD speeds (800 / 950 / 1050 rpm) with Hegman readings every 5 min, and two
solids levels. Plan them in days 60–90.

## 7. Dashboard on live data

`TSD_DATA_DIR=data/live streamlit run dashboard/app.py` after `export_historian.py`. Before
Step 4 has been re-run on real data the dashboard treats every batch as "test"; after it, the
split and features come from `data/processed_live/`. The simulated-data banner reads the folder
name, so it changes to the live source automatically.

## 8. 30 / 60 / 90-day plan for the first real batches

**Before day 0 (commissioning):**
- DCS I/O list → fill `tag_map.csv` NodeIds; MES phase names → `step_map.csv`.
- Instrument checks: RTD pair matched (ice-point and 40 °C comparison), VFD power parameter
  exposed, load cells tared, flow meter zero.
- Factory acceptance of the twin against `mock_plc_opcua.py` with the real tag names.
- OPC UA user / certificate for the twin PC; firewall rule OT → twin PC (port 4840 inbound to
  the DCS server only).

**Days 1–30 — watch, don't trust:**
- Twin runs in *shadow mode*: historian on, dashboard visible to the process engineer only, no
  alerts to operators.
- Every batch: tag-quality log reviewed (missing/stale tags), now-cast gap checked (a steady
  gap ≠ 0 points at an RTD offset, not at the model).
- After ~20 batches: recalibration steps 1–2; after ~30 batches with lab data: steps 3–4; re-run
  the pipeline. Acceptance before leaving shadow mode: physics-only MAE ≤ 1.5 °C on the last 10
  batches, hybrid now-cast ≤ 0.5 °C.
- Record every maintenance event (blade change, jacket clean, strainer) with batch numbers —
  these become the labels for tuning the fault rules.

**Days 31–60 — alerts to the panel:**
- Temperature pre-alarm and low-flow / overcharge alerts go to operators (they are instrument-
  based and need no fault history). Target: every breach pre-alarmed ≥ 5 min ahead, ≤ 5 watch
  flags per 100 batches; review each watch flag with the shift.
- Re-run the runtime normal levels (`check_step7`) once the recalibrated physics has 30 batches.
- Energy league table by product reviewed weekly; first what-if-based recipe trial on one
  product with the process engineer present (change one lever only).

**Days 61–90 — machine-condition rules and the optimiser:**
- With the first blade inspection and jacket clean logged, run `check_step8` on real labels
  (even 2–3 events give a sanity check of the thresholds; ≥ 80 % detection is a 6-month target).
- Deliberate trials for the tip-speed curve and phi_max (section 6); refit; then let the
  optimiser propose settings — adopt only where the predicted saving holds on 3 consecutive
  batches.
- Decide the historian: keep SQLite or hand the canonical row to the site historian.
- Hand-over: rule cards in the control-room binder, the assumptions register reviewed line by
  line against the plant, `basis: assumption` rows re-labelled `calibrated` with the batch
  count.

## 9. What will go wrong, and the fix

| Symptom | Likely cause | Fix |
|---|---|---|
| Rows missing, quality log full | PLC heartbeat not incrementing or wrong NodeId | Check `plc_heartbeat` in `tag_map.csv`; the twin treats all PLC values as bad without it |
| Now-cast gap steady at +0.8 °C from the first minute | RTD offset or thermowell not immersed | Compare TI1201 with a reference; the twin is not the problem |
| BLADE_WEAR flags on every batch after a VFD firmware update | Power parameter changed scaling | Re-fit the power number (recalibration step 1) |
| JACKET_FOULING on hot days only | Jacket inlet cycling within the 5-min window | Raise `alerts.u_window_min` to 8–10 |
| Energy forecast always low | MES planned time not the time actually run | Expose the planned time tag; check `step_map.csv` |
| Twin alert 1–2 cycles late | Settle time too long or publishing interval too long | 0.2 s settle, 500 ms publishing |
