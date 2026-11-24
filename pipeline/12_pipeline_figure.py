import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "DejaVu Serif"

W, H = 7.6, 9.4
fig = plt.figure(figsize=(W, H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
COL = {"c": ("#2f4f7f", "#eef1f7"), "d": ("#2e6b45", "#eef5f0"), "v": ("#8b3a3a", "#f6efec"), "a": ("#5a4a8e", "#f1eef7")}
PW, PH = 3.4, 4.05
PX = {0: 0.2, 1: 4.0}; PY = {0: 5.1, 1: 0.5}
panels = [("c", "1  CORPUS CONSTRUCTION", 0, 0), ("d", "2  PARSING AND EXTRACTION", 1, 0),
          ("v", "3  INDEXING AND VERIFICATION", 0, 1), ("a", "4  ANALYSIS", 1, 1)]
FS_T, FS_B, FS_C = 10.0, 9.0, 8.8
M = 0.16; BW = PW - 2 * M

def box(x, y, h, title, lines, key):
    dark, _ = COL[key]
    ax.add_patch(FancyBboxPatch((x, y), BW, h, boxstyle="round,pad=0.02,rounding_size=0.07", fc=dark, ec="none"))
    ax.text(x + BW / 2, y + h - 0.16, title, ha="center", va="top", color="white", fontsize=FS_T, fontweight="bold")
    ax.text(x + BW / 2, y + h - 0.44, "\n".join(lines), ha="center", va="top", color="#eeecf5", fontsize=FS_B, linespacing=1.22)

def chip(x, y, text, key):
    dark, light = COL[key]; h = 0.36 if "\n" not in text else 0.58
    ax.add_patch(FancyBboxPatch((x, y), BW, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc="white", ec=dark, lw=1.0))
    ax.text(x + BW / 2, y + h / 2, text, ha="center", va="center", color=dark, fontsize=FS_C)
    return h

def arr(x1, y1, x2, y2, color="#555", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=color, linestyle=ls, shrinkA=1, shrinkB=1))

for key, title, cx, cy in panels:
    dark, light = COL[key]; x, y = PX[cx], PY[cy]
    ax.add_patch(FancyBboxPatch((x, y), PW, PH, boxstyle="round,pad=0.02,rounding_size=0.1", fc=light, ec="none"))
    ax.text(x + PW / 2, y + PH - 0.08, title, ha="center", va="top", color=dark, fontsize=9.6, fontweight="bold")

# Phase 1
x, top = PX[0] + M, PY[0] + PH - 0.42
box(x, top - 0.95, 0.95, "Stratified sampling", ["GitHub search: 4 star bands x 2 activity", "windows; 2,000 candidates, fixed seed"], "c")
arr(x + BW/2, top - 0.95, x + BW/2, top - 1.22)
box(x, top - 2.17, 0.95, "Filtering (13 filters)", ["language, non-project signals,", "project structure: 905 accepted"], "c")
arr(x + BW/2, top - 2.17, x + BW/2, top - 2.44)
chip(x, top - 3.02, "905 READMEs, default-branch trees,\n11,675 configuration files", "c")
arr(x + BW/2, top - 3.02, x + BW/2, top - 3.26)
chip(x, top - 3.62, "Artifact index (paths, Makefile targets, ...)", "c")

# Phase 2
x, top = PX[1] + M, PY[0] + PH - 0.42
box(x, top - 0.82, 0.82, "README parser", ["headings (10 section classes), fenced", "blocks, inline code, links, prose; in order"], "d")
arr(x + BW/2, top - 0.82, x + BW/2, top - 1.04)
box(x, top - 1.86, 0.82, "Claim extractor", ["37 a priori pattern types, 4 surface", "contexts: 7,992 candidate claims"], "d")
arr(x + BW/2, top - 1.86, x + BW/2, top - 2.08)
box(x, top - 2.90, 0.82, "Claim hygiene (9 filters)", ["placeholders, URL fragments, user paths,", "routes, model ids: 958 removed (12.0%)"], "d")
arr(x + BW/2, top - 2.90, x + BW/2, top - 3.12)
chip(x, top - 3.12 - 0.36, "7,034 claims from 763 repositories", "d")

# Phase 3
x, top = PX[0] + M, PY[1] + PH - 0.42
box(x, top - 1.20, 1.20, "Sequence-aware verifier", ["claims read in README order: working", "directory, clone context, artifacts created", "by earlier steps, run-time lexicon"], "v")
arr(x + BW/2, top - 1.20, x + BW/2, top - 1.47)
box(x, top - 2.42, 0.95, "Per-type verification rules", ["path, directory, Makefile target, manifest,", "compose file, test evidence (Table 10)"], "v")
arr(x + BW/2, top - 2.42, x + BW/2, top - 2.68)
chip(x, top - 3.26, "supported 92.0%, drifted 3.5%,\nuncertain 4.5%; families D1 to D8", "v")

# Phase 4
x, top = PX[1] + M, PY[1] + PH - 0.42
bh, gap = 0.62, 0.10; y = top - bh
box(x, y, bh, "Prevalence and structure", ["rates with cluster-bootstrap 95% CIs"], "a"); y -= bh + gap
box(x, y, bh, "Manual validation", ["2 coders, blind; development + evaluation set"], "a"); y -= bh + gap
box(x, y, bh, "Sensitivity analysis", ["3 label thresholds; context-free baseline"], "a"); y -= bh + gap
box(x, y, bh, "Persistence (RQ5)", ["random cohort, 300 commits, Kaplan-Meier"], "a"); y -= bh + gap
box(x, y, bh, "Repository-level models", ["logistic and negative binomial"], "a")

# cross-panel arrows
yr = PY[0] + PH - 0.83
arr(PX[0] + PW, yr, PX[1], yr, "#2f4f7f")                                            # READMEs -> parser
ax.text((PX[0] + PW + PX[1]) / 2, yr + 0.07, "READMEs", ha="center", va="bottom", fontsize=7.4, color="#2f4f7f", rotation=0)
gy = (PY[0] + PY[1] + PH) / 2                                                         # middle of the row gap
cx2 = PX[1] + BW / 2 + M; cx1 = PX[0] + BW / 2 + M
ax.plot([cx2, cx2, cx1 + 0.55, cx1 + 0.55], [PY[0], gy, gy, PY[1] + PH + 0.03], color="#2e6b45", lw=1.3)
arr(cx1 + 0.55, gy, cx1 + 0.55, PY[1] + PH + 0.02, "#2e6b45")                       # claims -> verifier
ax.text(cx1 + 0.62, gy + 0.02, "claims", ha="left", va="bottom", fontsize=7.4, color="#2e6b45")
arr(cx1 - 0.55, PY[0] + 0.03, cx1 - 0.55, PY[1] + PH + 0.02, "#2f4f7f", "--")         # index -> verifier
ax.text(cx1 - 0.62, gy + 0.02, "artifact index", ha="right", va="bottom", fontsize=7.4, color="#2f4f7f")
yl = PY[1] + PH - 3.05
arr(PX[0] + PW, yl, PX[1], yl, "#8b3a3a")                                            # labels -> analysis
ax.text((PX[0] + PW + PX[1]) / 2, yl + 0.07, "labels", ha="center", va="bottom", fontsize=7.4, color="#8b3a3a")
ax.text(W / 2, 0.2, "Solid arrows: data flow; dashed arrow: the artifact index consumed by the verifier.\nAll stages are deterministic and every intermediate file is released.",
        ha="center", va="center", fontsize=7.4, color="#444", linespacing=1.3)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/pipeline_v2.{ext}")
print("ok")
