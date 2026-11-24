#!/usr/bin/env python3
"""
18_verification_sensitivity.py

Sensitivity of prevalence to the verification criterion for the three externally
grounded command types whose argument could name a repository artifact (Section 4.4 of the
article): uvicorn_run, gunicorn_run, npm_run.
  - ASGI/WSGI application specs (module:attr) are resolved against the file tree
    (module path + .py, or package __init__.py, anywhere in the tree).
  - npm run <script> is resolved against the "scripts" block of every package.json
    in the repository (data/01_corpus/configs_pilot_v3.csv).
Reports the primary estimate, the estimate with unresolved specs counted as drifted
(strict), and the estimate with all 25 claims moved to uncertain.

Output: results/verification_sensitivity.txt
"""
import os, gzip, csv, json
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
lab = pd.read_csv(os.path.join(D, "03_labels", "drift_results_detailed.csv")); k = lab[lab.status == "kept"].copy()
paths = {}
with gzip.open(os.path.join(D, "01_corpus", "trees_pilot_v3.csv.gz"), "rt") as f:
    for r in csv.DictReader(f): paths.setdefault(r["full_name"], set()).add(r["path"])
def module_exists(repo, spec):
    rel = str(spec).split(":")[0].replace(".", "/"); P = paths.get(repo, set())
    return any(p == rel + ".py" or p == rel + "/__init__.py" or p.endswith("/" + rel + ".py") or p.endswith("/" + rel + "/__init__.py") for p in P)
cfg = pd.read_csv(os.path.join(D, "01_corpus", "configs_pilot_v3.csv")); pj = cfg[cfg.basename == "package.json"]
def script_defined(repo, name):
    for c in pj[pj.full_name == repo].content:
        try:
            if name in (json.loads(c).get("scripts") or {}): return True
        except Exception: pass
    return False

lines = []; unresolved = 0; checked = 0; noise = 0
for r in k[k.claim_type.isin(["uvicorn_run", "gunicorn_run"])].itertuples():
    if ":" not in str(r.artifact_or_path): noise += 1; lines.append(f"{r.claim_type:13s} {r.full_name:40s} not an app spec (extraction noise): {str(r.snippet)[:60]}"); continue
    ok = module_exists(r.full_name, r.artifact_or_path); checked += 1; unresolved += (not ok)
    lines.append(f"{r.claim_type:13s} {r.full_name:40s} {r.artifact_or_path:40s} {'resolves' if ok else 'UNRESOLVED'}")
for r in k[k.claim_type == "npm_run"].itertuples():
    ok = script_defined(r.full_name, r.artifact_or_path); checked += 1; unresolved += (not ok)
    lines.append(f"{'npm_run':13s} {r.full_name:40s} {r.artifact_or_path:40s} {'defined in package.json' if ok else 'UNDEFINED'}")
n = len(k); drifted = int(k.drifted_v4.sum()); n_types = int(k.claim_type.isin(["uvicorn_run", "gunicorn_run", "npm_run"]).sum())
summary = [f"claims in the three types: {n_types} ({checked} checked, {noise} extraction noise)",
           f"unresolved against tree / package.json: {unresolved}",
           f"primary estimate: {drifted}/{n} = {100 * drifted / n:.2f}%",
           f"strict (unresolved counted as drifted): {drifted + unresolved}/{n} = {100 * (drifted + unresolved) / n:.2f}%",
           f"all {n_types} moved to uncertain: {drifted}/{n - n_types} = {100 * drifted / (n - n_types):.2f}%"]
txt = "\n".join(lines) + "\n\n" + "\n".join(summary) + "\n"
open(os.path.join(OUT, "verification_sensitivity.txt"), "w").write(txt); print(txt)
