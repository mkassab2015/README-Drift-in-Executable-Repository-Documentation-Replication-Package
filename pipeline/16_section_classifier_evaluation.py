#!/usr/bin/env python3
"""
16_section_classifier_evaluation.py

Scores the hand coding of 400 README headings (40 per predicted class) against the
keyword section classifier of Script 02 (Section 3.4 of the article).

Inputs (data/05_validation/section_classifier/):
  section_class_coding_workbook.xlsx   sheet 'Coding' (400 rows, column coder_class) and
                                       sheet 'Recode_100' (100 rows, blind re-code or second coder)
  section_class_key.csv                id -> classifier label (section_class), repository, heading
Outputs (same folder):
  section_classifier_scores.txt, section_classifier_confusion.csv, section_classifier_per_class.csv

Metrics: per-class precision, recall and F1 of the classifier against the hand labels
(hand label = reference); overall accuracy both unweighted (equal classes) and weighted
to the corpus frequency of each predicted class, since the sample is stratified on the
classifier's output; Cohen's kappa between the two hand codings of the 100-item subset.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "data", "05_validation", "section_classifier")
WB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "section_class_coding_workbook.xlsx")
CLASSES = ["installation", "usage", "run", "test", "build", "development", "deployment", "ci", "documentation", "other"]
# corpus frequency of each classifier label (headings, Script 02 output)
CORPUS_FREQ = {"other": 7254, "installation": 1574, "usage": 1024, "development": 482, "documentation": 462,
               "run": 372, "test": 246, "deployment": 195, "build": 139, "ci": 67}

key = pd.read_csv(os.path.join(D, "section_class_key.csv"))
cod = pd.read_excel(WB, sheet_name="Coding")[["id", "coder_class"]]
rec = pd.read_excel(WB, sheet_name="Recode_100")[["id", "coder_class"]].rename(columns={"coder_class": "recode_class"})
df = key.merge(cod, on="id").merge(rec, on="id", how="left")
df["coder_class"] = df.coder_class.astype(str).str.strip().str.lower()
missing = df[~df.coder_class.isin(CLASSES)]
if len(missing): sys.exit(f"{len(missing)} rows without a valid coder_class: {missing.id.tolist()[:10]} ...")

def kappa(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in CLASSES)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

lines = []
# agreement on the 100-item subset
sub = df.dropna(subset=["recode_class"]).copy(); sub["recode_class"] = sub.recode_class.astype(str).str.strip().str.lower()
if len(sub):
    lines.append(f"Recode subset: n={len(sub)}, raw agreement {(sub.coder_class == sub.recode_class).mean():.2f}, "
                 f"Cohen's kappa {kappa(sub.coder_class, sub.recode_class):.2f}")
# confusion: rows = hand label (reference), columns = classifier
conf = pd.crosstab(df.coder_class, df.section_class).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
conf.to_csv(os.path.join(D, "section_classifier_confusion.csv"))
rows = []
for c in CLASSES:
    tp = conf.loc[c, c]; fp = conf[c].sum() - tp; fn = conf.loc[c].sum() - tp
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    rows.append({"class": c, "n_predicted": int(tp + fp), "n_hand": int(tp + fn), "precision": round(p, 2), "recall_in_sample": round(r, 2), "f1_in_sample": round(f, 2)})
per = pd.DataFrame(rows)
# weighted accuracy: precision of each predicted class weighted by its corpus frequency
w = pd.Series(CORPUS_FREQ); w = w / w.sum()
acc_unw = np.mean([conf.loc[c, c] / conf[c].sum() for c in CLASSES])
acc_w = sum(w[c] * conf.loc[c, c] / conf[c].sum() for c in CLASSES)
lines += [f"Classifier accuracy (share of headings whose classifier label matches the hand label):",
          f"  unweighted over the ten classes {acc_unw:.2f}; weighted to corpus class frequencies {acc_w:.2f}",
          f"  functional classes only (excluding 'other'), weighted: "
          f"{sum(w[c] * conf.loc[c, c] / conf[c].sum() for c in CLASSES if c != 'other') / sum(w[c] for c in CLASSES if c != 'other'):.2f}",
          "Note: recall and F1 are within the stratified sample (40 predicted per class); since 'other' is 61% of corpus",
          "headings, functional headings hidden in 'other' are under-represented, so sample recall is an upper bound",
          "for corpus recall. The confusion row 'other' shows how many predicted-'other' headings were functional."]
per.to_csv(os.path.join(D, "section_classifier_per_class.csv"), index=False)
out = "\n".join(lines) + "\n\n" + per.to_string(index=False) + "\n\nConfusion (rows: hand label; columns: classifier):\n" + conf.to_string() + "\n"
open(os.path.join(D, "section_classifier_scores.txt"), "w").write(out); print(out)

# ------------------------------------------------------------------ admission sensitivity (Section 4.4, "Section admission")
try:
    lab = pd.read_csv(os.path.join(ROOT, "data", "03_labels", "drift_results_detailed.csv"))
    lab = lab[lab.status == "kept"].copy()
    cand = pd.read_csv(os.path.join(ROOT, "data", "02_claims", "candidate_claims_pilot_v3.csv"))[
        ["full_name", "line_no", "snippet", "claim_type", "operational_hint"]].drop_duplicates(["full_name", "line_no", "snippet", "claim_type"])
    lab = lab.merge(cand, on=["full_name", "line_no", "snippet", "claim_type"], how="left")
    lab["drift"] = lab.drifted_v4.astype(bool)
    ar = lab[lab.claim_family == "artifact_reference"]; cmd = lab[lab.claim_family == "command"]
    precise = ["installation", "usage", "run", "development", "documentation"]
    out2 = ["\nAdmission sensitivity (artifact references restricted; commands unchanged):"]
    for name, mask in [("all admitted", ar.section_class.notna()), ("precise classes only", ar.section_class.isin(precise)),
                       ("operational hint only", ar.operational_hint == 1), ("precise classes or hint", ar.section_class.isin(precise) | (ar.operational_hint == 1))]:
        a = ar[mask]; tot = len(a) + len(cmd); dr = a.drift.sum() + cmd.drift.sum()
        out2.append(f"  {name:26s} artifact refs {len(a):5d}, drifted {a.drift.sum():3d} ({100 * a.drift.sum() / len(a):.1f}%); corpus {dr}/{tot} = {100 * dr / tot:.2f}%")
    open(os.path.join(D, "section_classifier_scores.txt"), "a").write("\n".join(out2) + "\n"); print("\n".join(out2))
except FileNotFoundError as e:
    print("admission sensitivity skipped:", e)
