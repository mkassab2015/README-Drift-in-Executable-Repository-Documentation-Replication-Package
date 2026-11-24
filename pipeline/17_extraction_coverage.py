#!/usr/bin/env python3
"""
17_extraction_coverage.py

Coverage account of the claim extractor (Section 3.4 of the article): for every README element
produced by Script 02, whether Script 03 extracted at least one candidate claim from
it, and a lexical classification of the elements it did not extract, so that the
part of the unextracted mass that lies inside the study's construct (commands and
repository-artifact references in operational sections) can be bounded.

Inputs:  data/02_claims/readme_structure_pilot_v3.csv, data/02_claims/candidate_claims_pilot_v3.csv
Outputs: results/extraction_coverage/coverage_by_channel.csv, unextracted_shell_blocks.csv,
         unextracted_inline_code.csv, unextracted_local_links.csv, summary.txt
"""
import os, re
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "extraction_coverage"); os.makedirs(OUT, exist_ok=True)
s = pd.read_csv(os.path.join(ROOT, "data", "02_claims", "readme_structure_pilot_v3.csv"))
c = pd.read_csv(os.path.join(ROOT, "data", "02_claims", "candidate_claims_pilot_v3.csv"))
keys = set(zip(c.full_name, c.line_no))
s["extracted"] = [(a, b) in keys for a, b in zip(s.full_name, s.line_no)]
OPER = {"installation", "usage", "run", "test", "build", "development", "deployment", "ci"}
s["admitted"] = s.section_class.isin(OPER) | (s.operational_hint == 1)     # admission condition for artifact references
SHELL = {"bash", "sh", "shell", "console", "zsh", "commandline", "powershell", "cmd", "bat", "terminal"}
PY = {"python", "py", "python3", "pycon"}

rows, lines = [], []
def add(channel, sub, df):
    rows.append({"channel": channel, "subset": sub, "elements": len(df), "extracted": int(df.extracted.sum()),
                 "share_extracted": round(df.extracted.mean(), 3) if len(df) else None})

# ---------------------------------------------------------------- fenced blocks
f = s[s.elem_type == "fenced_code"].copy(); f["lang"] = f.code_lang.fillna("").str.lower()
f["group"] = f.lang.map(lambda l: "shell-tagged" if l in SHELL else "python-tagged" if l in PY else "untagged" if l == "" else "other language")
add("fenced block", "all", f)
for g, d in f.groupby("group"): add("fenced block", g, d)
VOCAB = r"^(pip3?|python3?|git|docker|docker-compose|npm|npx|yarn|make|cd|conda|poetry|uv|bash|sh|source|curl|wget|export|streamlit|pytest|tox|gunicorn|uvicorn|flask|django-admin|mkdir|cp|mv)\b"
SYS = r"^(pipx|uvx|brew|apt|apt-get|dnf|yum|scoop|choco|winget|snap|cargo|go|java|node|deno|ruby|gem|php|composer|helm|kubectl|aws|gcloud|az|terraform|ansible|systemctl|sudo|chmod|ls|cat|echo|rm|ln|tar|unzip|ssh|scp|ollama)\b"
def shell_kind(block):
    ls = [l.strip() for l in str(block).split("\n") if l.strip()]
    cmd = [l for l in ls if not l.startswith("#")]
    if not ls: return "empty"
    if not cmd: return "comments only"
    if all(re.match(r'^[\$>]?\s*\{|^[\[\]{}",]|^[A-Za-z_]+\s*[:=]', l) for l in cmd): return "configuration or data"
    if any(re.match(r"^(\$|>|\(\w+\)\$|[\w@\-]+:[\w~/]*\$)\s", l) for l in cmd): return "prompt-prefixed command"
    if all(re.match(VOCAB, l) for l in cmd): return "vocabulary command not matched"
    if any(re.match(SYS, l) for l in cmd): return "other package manager or system tool"
    if any(re.match(r"^\./", l) or re.match(r"^[\w./\-]+\.(sh|py)\b", l) for l in cmd): return "local script invocation"
    if any(re.match(r"^[a-z][\w\-]*(\s|$)", l) for l in cmd): return "project-specific executable"
    return "output, URL or other"
sb = f[(f.group == "shell-tagged") & ~f.extracted].copy(); sb["kind"] = sb.content.map(shell_kind)
sb[["full_name", "line_no", "section_class", "admitted", "kind", "content"]].to_csv(os.path.join(OUT, "unextracted_shell_blocks.csv"), index=False)
lines.append("Unextracted shell-tagged fenced blocks by kind:\n" + sb.kind.value_counts().to_string())

# ---------------------------------------------------------------- inline code
TOK = re.compile(r"(?<![\w@:])(?:\.{1,2}/)?[\w.\-]+(?:/[\w.\-]+)+/?|(?<![\w/.])[\w\-]+\.(?:py|sh|yml|yaml|json|toml|cfg|ini|txt|md|csv|ipynb)\b")
def inline_kind(t):
    t = str(t).strip()
    if "http" in t or "://" in t: return "URL"
    if TOK.search(t) and not re.search(r"^\d+(\.\d+)+$", t): return "path-like"
    if re.match(r"^-{1,2}\w", t): return "option or flag"
    if re.match(r"^[A-Z][A-Z0-9_]{2,}$", t): return "environment variable or constant"
    if re.match(VOCAB, t): return "vocabulary command not matched"
    if re.match(r"^[\w.]+\(.*\)$", t) or ("." in t and " " not in t): return "identifier"
    return "word, value or name"
ic = s[s.elem_type == "inline_code"].copy(); add("inline code", "all", ic)
ui = ic[~ic.extracted].copy(); ui["kind"] = ui.content.map(inline_kind)
ui[["full_name", "line_no", "section_class", "admitted", "kind", "content"]].to_csv(os.path.join(OUT, "unextracted_inline_code.csv"), index=False)
lines.append("\nUnextracted inline code by kind:\n" + ui.kind.value_counts().to_string())
pl = ui[ui.kind == "path-like"]
lines.append(f"path-like inline spans not extracted: {len(pl)}, of which in admitted sections (inside the construct): {int(pl.admitted.sum())}")

# ---------------------------------------------------------------- links
lk = s[s.elem_type == "markdown_link"].copy()
add("markdown link", "external target", lk[lk.link_target_type == "external"])
loc = lk[lk.link_target_type.str.startswith("local", na=False)]
add("markdown link", "local target, all", loc)
add("markdown link", "local target, admitted section", loc[loc.admitted])
add("markdown link", "local target, non-admitted section", loc[~loc.admitted])
ul = loc[loc.admitted & ~loc.extracted]
ul[["full_name", "line_no", "section_class", "link_target_type", "link_text", "link_target"]].to_csv(os.path.join(OUT, "unextracted_local_links.csv"), index=False)
lines.append("\nLocal links in admitted sections not extracted, by target type:\n" + ul.link_target_type.value_counts().to_string())

# ---------------------------------------------------------------- prose
add("prose line", "all", s[s.elem_type == "section_line"])

cov = pd.DataFrame(rows); cov.to_csv(os.path.join(OUT, "coverage_by_channel.csv"), index=False)
inside = {"local script invocation", "vocabulary command not matched", "prompt-prefixed command"}
n_inside = int(sb.kind.isin(inside).sum()) + int(pl.admitted.sum()) + len(ul)
n_exec = int((sb.kind == "project-specific executable").sum())
lines.append(f"\nElements inside the construct that were not extracted (upper bound): shell blocks {int(sb.kind.isin(inside).sum())} "
             f"+ path-like inline spans in admitted sections {int(pl.admitted.sum())} + local links in admitted sections {len(ul)} = {n_inside} "
             f"({100 * n_inside / len(c):.1f}% of the {len(c)} candidates)")
lines.append(f"Project-specific executables in unextracted shell blocks (externally grounded commands, not verifiable against the tree): {n_exec}; "
             f"counting them as supported claims would give 246/{7034 + n_exec} = {100 * 246 / (7034 + n_exec):.2f}%")
txt = cov.to_string(index=False) + "\n\n" + "\n".join(lines) + "\n"
open(os.path.join(OUT, "summary.txt"), "w").write(txt); print(txt)
