"""
common.py - shared loading, caching and styling for the dashboard pages.

Everything heavy is cached once per session: the measured time series, the batch table,
the models, the what-if twin, the runtime baselines, and the fleet-level runtime results
(precomputed by scripts/check_step7.py into data/processed/, recomputed here if missing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402
DATA_DIR = Path(os.environ.get("TSD_DATA_DIR", ROOT / "data" / "raw"))        # data/raw (simulated) or data/live (plant)
PROC_DIR = ROOT / "data" / ("processed" if DATA_DIR.name == "raw" else f"processed_{DATA_DIR.name}")

from tsd_twin import models as md, runtime as rt, whatif as wi  # noqa: E402
from tsd_twin.config import get, load_config  # noqa: E402

NAVY, ORANGE, GREEN, GREY, RED, PURPLE, BLUE = "#1F3864", "#C55A11", "#548235", "#8A8A8A", "#C00000", "#7030A0", "#2E75B6"
PHASE_LABEL = {"charge_liquids": "Charge", "pigment_addition": "Pigment", "high_speed_dispersion": "High-speed dispersion", "letdown": "Letdown"}

CSS = """
<style>
html, body, [class*="css"], .stApp { font-family: Arial, Helvetica, sans-serif; }
.stApp { background: #F4F6F9; }
section[data-testid="stSidebar"] { background: #1F3864; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * { color: #1F3864 !important; }
section[data-testid="stSidebar"] input { color: #1F3864 !important; }
.tsd-header { background: #1F3864; color: #FFFFFF; padding: 10px 18px; border-radius: 6px; margin-bottom: 12px;
              display: flex; justify-content: space-between; align-items: center; }
.tsd-header .title { font-size: 20px; font-weight: 700; letter-spacing: 0.3px; }
.tsd-header .sub { font-size: 12px; opacity: 0.85; }
.tsd-banner { background: #FFF2CC; border-left: 5px solid #C55A11; padding: 6px 12px; font-size: 12px; margin-bottom: 10px; color: #333; }
.tile { background: #FFFFFF; border: 1px solid #DCE6F1; border-left: 5px solid #1F3864; border-radius: 6px; padding: 8px 12px; min-height: 76px; }
.tile .label { font-size: 11px; color: #5A6B7B; text-transform: uppercase; letter-spacing: 0.5px; }
.tile .value { font-size: 24px; font-weight: 700; color: #1F3864; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tile .value.text { font-size: 16px; padding-top: 4px; }
.tile .unit { font-size: 12px; color: #5A6B7B; margin-left: 3px; }
.tile .note { font-size: 11px; color: #5A6B7B; }
.tile.warn { border-left-color: #C55A11; } .tile.warn .value { color: #C55A11; }
.tile.alarm { border-left-color: #C00000; background: #FDECEC; } .tile.alarm .value { color: #C00000; }
.tile.ok { border-left-color: #548235; } .tile.ok .value { color: #548235; }
.alert-strip { border-radius: 6px; padding: 10px 14px; font-size: 14px; font-weight: 600; margin: 6px 0 10px 0; }
.alert-none { background: #E2EFDA; color: #385723; border-left: 6px solid #548235; }
.alert-warn { background: #FFF2CC; color: #7F4A00; border-left: 6px solid #C55A11; }
.alert-alarm { background: #FDECEC; color: #8B0000; border-left: 6px solid #C00000; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #DCE6F1; border-radius: 6px; padding: 6px 10px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="tsd-header"><div><div class="title">{title}</div><div class="sub">{subtitle}</div></div>'
                f'<div class="sub">TSD digital twin &middot; 3,000 L twin-shaft disperser &middot; SI units</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tsd-banner">SIMULATED DATA - every number on this page comes from the Step 3 simulator and the '
                'models trained on it. Nothing here is plant performance until the Step 9 cut-over.</div>', unsafe_allow_html=True)


def tile(label: str, value, unit: str = "", note: str = "", state: str = "") -> str:
    v = "-" if value is None or (isinstance(value, float) and pd.isna(value)) else value
    cls = "value text" if isinstance(v, str) and len(v) > 6 and not v.replace(".", "").replace("-", "").isdigit() else "value"
    return (f'<div class="tile {state}"><div class="label">{label}</div><div class="{cls}">{v}<span class="unit">{unit}</span></div>'
            f'<div class="note">{note}</div></div>')


def tiles(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(tile(*it), unsafe_allow_html=True)


def plot_layout(fig, height=360, title=None):
    fig.update_layout(template="plotly_white", font=dict(family="Arial", size=12), height=height, margin=dict(l=50, r=20, t=40 if title else 20, b=40),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"), title=title, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    fig.update_xaxes(gridcolor="#E8EDF3", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EDF3", zeroline=False)
    return fig


def add_phase_bands(fig, run: pd.DataFrame, row=None, col=None) -> None:
    for step, grp in run.groupby("step", sort=False):
        x0, x1 = grp["t_min"].iloc[0], grp["t_min"].iloc[-1] + 1 / 6
        shade = 0.07 if step in ("pigment_addition", "letdown") else 0.0
        kw = dict(row=row, col=col) if row else {}
        fig.add_vrect(x0=x0, x1=x1, fillcolor=NAVY, opacity=shade, line_width=0, annotation_text=PHASE_LABEL[step],
                      annotation_position="top left", annotation_font_size=10, annotation_font_color=GREY, **kw)


# --------------------------------------------------------------------- data --
@st.cache_resource(show_spinner="Loading the twin (config, data, models)...")
def load_core() -> dict:
    cfg = load_config()
    if not (DATA_DIR / "timeseries.csv").exists() and DATA_DIR.name == "raw":
        # fresh clone on a host (Render, Streamlit Cloud, a new laptop): regenerate the simulated year (~20 s, deterministic)
        from tsd_twin.simulate import generate_dataset
        with st.spinner("First start: generating the simulated plant year (about 20 s)..."):
            generate_dataset(cfg, DATA_DIR, verbose=False)
    ts = pd.read_csv(DATA_DIR / "timeseries.csv", parse_dates=["timestamp"])
    batches = pd.read_csv(DATA_DIR / "batches.csv", parse_dates=["start_time"]).set_index("batch_id")
    if (PROC_DIR / "split.csv").exists():
        split = pd.read_csv(PROC_DIR / "split.csv").set_index("batch_id")
    else:                                                    # live data before Step 4 has been re-run: everything is 'test'
        split = pd.DataFrame({"split_random": "test", "split_time": "test", "fold": 0}, index=batches.index)
    bf = pd.read_csv(PROC_DIR / "batch_features.csv").set_index("batch_id") if (PROC_DIR / "batch_features.csv").exists() else pd.DataFrame(index=batches.index)
    rec_path = DATA_DIR / "recipes.csv" if (DATA_DIR / "recipes.csv").exists() else ROOT / "data" / "raw" / "recipes.csv"
    recipes = pd.read_csv(rec_path).set_index("recipe_id")
    tm, bm, card = md.load_models(ROOT / "models")
    twin = wi.load_twin(ROOT / "models")
    baselines = rt.Baselines.load(ROOT / "models" / "runtime_calibration.json")
    thr = rt.Thresholds.from_config(cfg)
    return dict(cfg=cfg, ts=ts, batches=batches, split=split, bf=bf, recipes=recipes, tm=tm, bm=bm, card=card,
                twin=twin, baselines=baselines, thr=thr)


@st.cache_data(show_spinner=False)
def batch_run(batch_id: str) -> pd.DataFrame:
    core = load_core()
    b = core["ts"][core["ts"]["batch_id"] == batch_id]
    meta = core["batches"].loc[batch_id]
    return rt.prepare_batch(b, core["cfg"], core["tm"], float(meta["batch_mass_nominal_kg"]), float(meta["cw_flow_setpoint_m3h"]),
                            core["thr"], core["baselines"])


@st.cache_data(show_spinner="Running the twin over every batch (first time only)...")
def fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-batch verdicts and the alert episodes for the whole history."""
    core = load_core()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    fv, fa = PROC_DIR / "fleet_verdicts.csv", PROC_DIR / "fleet_alerts.csv"
    if fv.exists() and fa.exists():
        return pd.read_csv(fv, parse_dates=["start_time"]), pd.read_csv(fa)
    verdicts, alerts = [], []
    for bid in core["batches"].index:
        run = batch_run(bid)
        verdicts.append(rt.batch_verdict(run))
        alerts.append(rt.alert_table(run))
    V = pd.DataFrame(verdicts).set_index("batch_id").join(core["batches"][["start_time", "recipe_id", "recipe_name", "product_family", "batch_mass_kg",
                                                                            "dispersion_min", "n_extensions", "hegman_reached_lab", "fault_injected"]]).reset_index()
    A = pd.concat([a for a in alerts if not a.empty], ignore_index=True) if any(not a.empty for a in alerts) else pd.DataFrame()
    V.to_csv(fv, index=False)
    A.to_csv(fa, index=False)
    return V, A


def default_batch() -> str:
    core = load_core()
    te = core["split"][core["split"]["split_random"] == "test"].index
    br = core["batches"].loc[te]
    br = br[br["breached"]]
    return br.index[0] if len(br) else te[0]
