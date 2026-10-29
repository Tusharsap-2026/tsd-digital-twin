"""
config.py - single point of access to config/assumptions.yaml

Rule of this project: every assumed number lives in ONE file. This module
loads that file, checks each parameter is well-formed (value / unit / basis),
and offers three helpers that every later step uses:

    cfg = load_config()
    get(cfg, "equipment.hsd.motor_rated_kw")   -> 55
    parameter_table(cfg)                        -> DataFrame: the assumptions register
    assumptions_only(cfg)                       -> DataFrame: only the rows that must be confirmed
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]          # .../tsd_twin
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
VALID_BASIS = {"given", "assumption", "physical"}


def is_param(node: Any) -> bool:
    """A parameter leaf is a dict carrying at least 'value' and 'basis'."""
    return isinstance(node, dict) and "value" in node and "basis" in node


def walk(node: Any, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (dotted_path, parameter_dict) for every parameter in the tree."""
    if is_param(node):
        yield path, node
    elif isinstance(node, dict):
        for key, child in node.items():
            yield from walk(child, f"{path}.{key}" if path else key)


def validate(cfg: dict) -> list[str]:
    """Return a list of problems. An empty list means the config is well-formed."""
    problems: list[str] = []
    for path, p in walk(cfg):
        if p["basis"] not in VALID_BASIS:
            problems.append(f"{path}: basis '{p['basis']}' must be one of {sorted(VALID_BASIS)}")
        if "unit" not in p:
            problems.append(f"{path}: missing 'unit'")
        v = p["value"]
        if (isinstance(v, list) and len(v) == 2
                and all(isinstance(x, (int, float)) for x in v) and v[0] > v[1]):
            problems.append(f"{path}: range {v} has min > max")
    return problems


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load and validate the assumptions file. Raises if anything is malformed."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    problems = validate(cfg)
    if problems:
        raise ValueError("assumptions.yaml has problems:\n  - " + "\n  - ".join(problems))
    return cfg


def get(cfg: dict, dotted_path: str) -> Any:
    """Value at a dotted path: the raw 'value' for a parameter, or the sub-tree otherwise."""
    node: Any = cfg
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"'{dotted_path}' not found in assumptions.yaml (failed at '{key}')")
        node = node[key]
    return node["value"] if is_param(node) else node


def parameter_table(cfg: dict) -> pd.DataFrame:
    """Every parameter as one row: the assumptions register."""
    rows = [
        {"parameter": path, "value": p["value"], "unit": p.get("unit", "-"),
         "basis": p["basis"], "note": p.get("note", "")}
        for path, p in walk(cfg)
    ]
    return pd.DataFrame(rows)


def assumptions_only(cfg: dict) -> pd.DataFrame:
    """Only the rows with basis = assumption: the list to confirm with vendors / first batches."""
    t = parameter_table(cfg)
    return t[t["basis"] == "assumption"].reset_index(drop=True)


def update_values(path: Path | str, updates: dict) -> list[str]:
    """Update 'value' (and optionally 'note') of parameters in assumptions.yaml IN PLACE, keeping
    every comment and the file layout. updates: {"physics.hsd_power.np_turbulent": (0.324, "note"), ...}
    Returns the keys that were updated."""
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    # map every dotted key to the line index of its "key:" line, using indentation
    stack: list[tuple[int, str]] = []
    key_lines: dict[str, int] = {}
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        key_lines[".".join(k for _, k in stack)] = i
    changed = []
    for dotted, (value, note) in updates.items():
        if dotted not in key_lines:
            continue
        start = key_lines[dotted]
        indent = len(lines[start]) - len(lines[start].lstrip(" "))
        j = start + 1
        has_note = False
        while j < len(lines) and (len(lines[j]) - len(lines[j].lstrip(" ")) > indent or not lines[j].strip()):
            l = lines[j]
            li = len(l) - len(l.lstrip(" "))
            if l.strip().startswith("value:") and li == indent + 2:
                v = f'"{value}"' if isinstance(value, str) else (f"{value:.6g}" if isinstance(value, float) else str(value))
                lines[j] = " " * li + f"value: {v}\n"
            if l.strip().startswith("note:") and li == indent + 2 and note is not None:
                lines[j] = " " * li + f'note: "{note}"\n'
                has_note = True
            j += 1
        if note is not None and not has_note:
            lines.insert(j, " " * (indent + 2) + f'note: "{note}"\n')
            for k2, v2 in key_lines.items():
                if v2 > start:
                    key_lines[k2] = v2 + 1
        changed.append(dotted)
    path.write_text("".join(lines))
    return changed
