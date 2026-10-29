# TSD Digital Twin — prototype

Digital twin of a twin-shaft paint disperser (3,000 L vessel, 55 kW high-speed
disperser, 11 kW anchor, chilled-water jacket). Runs on **simulated** data until
the plant's PLC/DCS tags are connected (Step 9).

> Nothing produced by this prototype is real plant performance until the
> `meta.data_status` line in `config/assumptions.yaml` says otherwise.

## Quick start (no developer tools needed)

1. Install Python 3.10 or newer from https://www.python.org — on Windows tick **"Add python to PATH"**.
2. Unzip the project anywhere (e.g. `C:\tsd_twin`).
3. Windows: double-click **`start_dashboard.bat`**. Linux/macOS: `bash start_dashboard.sh`.
   The first run creates a private environment, installs the packages (2–5 min) and
   generates the simulated data (20 s). Every later run just starts the app.
4. The browser opens at http://localhost:8501. Close the black window to stop.

## Folder structure

```
tsd_twin/
├── README.md
├── requirements.txt
├── start_dashboard.bat         <- Windows one-click start
├── start_dashboard.sh          <- Linux / macOS one-click start
├── config/
│   ├── assumptions.yaml        <- THE one file holding every number (given / assumption / physical / calibrated)
│   ├── tag_map.csv             <- Step 9: canonical column <-> PLC/MES/LAB tag (NodeId, topic, units, deadband)
│   └── step_map.csv            <- Step 9: MES phase names -> the four canonical steps
├── docs/
│   └── PATH_TO_LIVE.md         <- Step 9: architecture, protocols, instruments, historian, MES, recalibration, 30/60/90 plan
├── src/tsd_twin/
│   ├── __init__.py
│   ├── config.py               <- loads + validates assumptions.yaml, prints the register
│   ├── schema.py               <- canonical column names = the twin's tag list
│   ├── kpis.py                 <- every KPI formula (process KPIs and model KPIs)
│   ├── physics.py              <- Step 2: power, viscosity, energy balance, grind, drive limit
│   ├── simulate.py             <- Step 3: recipe catalogue, climate, operator, lots, noise, faults
│   ├── features.py             <- Step 4: physics replay baselines, row/batch features, split by batch
│   ├── models.py               <- Step 5: hybrid temperature models, energy / time-to-Hegman / peak models
│   ├── whatif.py               <- Step 6: Settings -> run_scenario / compare_scenarios / optimise
│   ├── runtime.py              <- Step 7: per-batch engine - now-cast, 5-min-ahead, breach ETA, fault rules, alert log
│   ├── faults.py               <- Step 8: rule statistics, threshold sweeps, scoring, drift tracker, rule cards
│   └── live.py                 <- Step 9: TagMap, TagBuffer, BatchAssembler, Replay/OpcUa/Mqtt sources, Historian, LiveTwin
├── scripts/
│   ├── check_step1.py          <- Step 1 proof
│   ├── check_step2.py          <- Step 2 proof (reference batches, closure, sensitivity, plot)
│   ├── check_step3.py          <- Step 3: generates data/raw/ and validates it
│   ├── check_step4.py          <- Step 4: builds data/processed/ and audits it
│   ├── check_step5.py          <- Step 5: trains, scores on unseen batches, saves models/
│   ├── check_step6.py          <- Step 6: calibrates, validates the what-if on test batches, runs the optimiser
│   ├── whatif_cli.py           <- Step 6: what-if from the command line
│   ├── check_step7.py          <- Step 7: runtime over the whole history + headless dashboard smoke test
│   ├── check_step8.py          <- Step 8: tunes and validates the fault rules, writes thresholds back to the config
│   ├── check_step9.py          <- Step 9: replay + OPC UA end to end, historian export, recalibration recovery
│   ├── run_live.py             <- Step 9: the twin on a live source (replay | opcua | mqtt)
│   ├── mock_plc_opcua.py       <- Step 9: OPC UA server that behaves like the plant (FAT, training)
│   ├── export_historian.py     <- Step 9: historian -> canonical files for the pipeline and dashboard
│   ├── recalibrate.py          <- Step 9: ordered refit of the physics on real batches
│   └── make_dataset.py         <- regenerate with another size / seed
├── dashboard/
│   ├── app.py                  <- Step 7: streamlit run dashboard/app.py
│   ├── common.py               <- loading, caching, styling
│   └── pages/                  1_Live_batch, 2_What_if, 3_Batch_history, 4_Alerts
├── data/
│   ├── raw/                    timeseries.csv, timeseries_truth.csv, batches.csv, recipes.csv (Step 3)
│   ├── processed/              row_features.csv.gz, batch_features.csv, split.csv, feature_dictionary.csv (Step 4)
│   └── dictionary/             data dictionary (written by check_step1.py)
├── models/                     temperature_models.joblib, batch_models.joblib, model_card.json (Step 5),
│                               whatif_calibration.json (Step 6)
└── reports/                    assumptions register, validation reports
```

Files marked (Step n) do not exist yet; each step adds its own.

## Run Step 1

```bash
cd tsd_twin
python -m venv .venv                 # once
.venv\Scripts\activate               # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt      # once
python scripts/check_step1.py
```

Expected: the assumptions register, the engineering sanity checks, the data
dictionary, KPI values for a hand-made toy batch, and the line
`ALL STEP 1 CHECKS PASSED`.

## Run Step 2

```bash
python scripts/check_step2.py
```

Expected: the physics parameter list, power-number and viscosity curves, the
power-vs-speed table showing the 55 kW drive limit, two reference batches with
phase tables, an energy-balance closure of 0.0000 %, a sensitivity table where
every disturbance moves the right way, and `reports/step2_reference_batches.html`
(open it in a browser; it is self-contained).

Using the physics model from your own code:

```python
from tsd_twin.config import load_config
from tsd_twin import physics as ph
cfg = load_config()
recipe = ph.Recipe("water_based", 3000, solids_pct=58, pvc_pct=55, target_hegman=6.0)
phases = ph.standard_phases(recipe, cfg, dispersion_rpm=950, dispersion_min=35)
df, summary = ph.simulate_batch(recipe, phases, ph.Conditions(T_cw_in_c=22, cw_flow_m3h=10), cfg)
```

## Run Step 3

```bash
python scripts/check_step3.py          # ~15 s: generates data/raw/ and validates it
python scripts/make_dataset.py --n 500 --seed 7 --out data/raw_seed7   # optional variants
```

`data/raw/timeseries.csv` holds only what the plant will measure (22 tags at 10 s,
sparse lab columns). `timeseries_truth.csv` and the right-hand half of `batches.csv`
hold the simulator's hidden states and the injected fault labels: use them to
validate the twin, never to train it.

## Run Step 4

```bash
python scripts/check_step4.py          # ~30 s: builds data/processed/ from data/raw/
```

Row features carry the physics baselines (`T_phys_open` with modelled power,
`T_phys_replay` with measured power) and the online `lag_*` columns; batch
features are what is known at the start of high-speed dispersion plus the
physics forecast for the planned schedule. `split.csv` assigns every batch to
train or test (random, stratified; and a time split) and to one of five folds.

## Run Step 5

```bash
python scripts/check_step5.py          # ~60 s: trains on the train batches, scores the unseen test batches
```

Prints the scorecards with verdicts against the KPI targets, retrains on the first
70 % of the year as a robustness check, writes `models/` and
`reports/step5_model_report.html`. Model inputs by predictor are listed at the top
of `src/tsd_twin/models.py`.

## Run Step 6

```bash
python scripts/check_step6.py          # ~60 s: calibration, validation, sweeps, optimiser, plot
python scripts/whatif_cli.py --rpm 950 --minutes 35 --cw-in 28 --t-start 33 --ambient 35 \
       --compare "rpm=900,minutes=40" "rpm=1000,cw_flow=15,minutes=30" --optimise --plot reports/my_whatif.html
```

From Python:

```python
from tsd_twin.config import load_config
from tsd_twin import whatif as wi
cfg, twin = load_config(), wi.load_twin("models")
s = wi.Settings("water_based", 3000, 58, 55, 6.0, hsd_rpm=950, anchor_rpm=35, dispersion_min=35,
                cw_inlet_c=28, cw_flow_m3h=10, T_start_c=33, T_ambient_c=35)
r = wi.run_scenario(s, cfg, twin)     # r.summary, r.curve, r.checks
best = wi.optimise(s, cfg, twin)      # best["explanation"], best["best"], best["candidates"]
```

## Run Step 7

```bash
python scripts/check_step7.py          # ~90 s: runtime calibration, fleet results, headless page screenshots
streamlit run dashboard/app.py         # then open http://localhost:8501
```

Pages: **Live batch** (replay any batch with the twin alongside: now-cast, 5-min-ahead,
breach ETA, alert strip with cause), **What-if** (sliders, three scenarios, optimiser),
**Batch history** (KPI trends, energy league table, fault flags, optional ground-truth
overlay), **Alerts** (alert log with lead times and causes, per-batch alert timeline).
The runtime's normal plant-vs-physics levels are in `models/runtime_calibration.json`;
alert thresholds are the `alerts` section of `config/assumptions.yaml`.

## Run Step 8

```bash
python scripts/check_step8.py          # ~3 min: sweeps, scoring, drift tracker, config write-back, dashboard precompute
```

Writes the tuned thresholds (with achieved detection / false-alarm rates in their notes)
into the `alerts` section of `config/assumptions.yaml`, the plain-language rule cards to
`reports/step8_rule_cards.md`, and `reports/step8_fault_detection.html`. The rules never
train on the fault labels; the labels only score them.

## Run Step 9

```bash
python scripts/check_step9.py                       # ~4 min: replay, OPC UA end to end, export, recalibration
python scripts/mock_plc_opcua.py --speed 20          # a plant-like OPC UA server (another terminal)
python scripts/run_live.py --source opcua --endpoint opc.tcp://127.0.0.1:4840/tsd/
python scripts/export_historian.py --out data/live   # historian -> canonical files
TSD_DATA_DIR=data/live streamlit run dashboard/app.py
python scripts/recalibrate.py --data data/live       # after ~30 real batches; then re-run Steps 4-8
```

`docs/PATH_TO_LIVE.md` is the hand-over document for the controls and MES teams.

## Scope of the twin (Step 1)

**Inside the boundary:** the product mass in the vessel, the HSD blade and
shaft, the anchor/scraper, the vessel wall and cooling jacket, and the step
sequence charge → pigment addition → high-speed dispersion → letdown.

**Outside (boundary conditions, taken as given):** the chiller plant (chilled
water inlet temperature and flow), raw-material dosing accuracy (batch mass is
taken from the load cells), the letdown/thinning tank, bead mills, the QC lab's
turnaround time, and any closed-loop control (the twin is advisory only).

**Inputs:** recipe attributes from MES, set-points from the PLC, utility
conditions, initial product temperature, and (later) lab Hegman/viscosity.

**Outputs:** predicted temperature curve, kWh and kWh/t, time to target Hegman,
limit-breach ETA with likely cause, and fault indicators (blade wear, jacket
fouling, low cooling flow, overcharge).

KPI definitions and formulas: `src/tsd_twin/kpis.py` (docstring at the top).
