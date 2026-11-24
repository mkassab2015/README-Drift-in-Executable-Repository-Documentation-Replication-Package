#!/usr/bin/env python3
"""
15_representativeness_analysis.py

Representativeness check of the corpus against its sampling frame (Section 3.3,
Tables "population" and "representativeness"; Section 6, "Sampling frame").

Inputs (data/01_corpus/representativeness/, produced by 13_* and 14_*):
  population_stratum_counts.csv  size of each of the eight strata (sharded enumeration)
  corpus905_activity.csv         contributors/commits/releases/licence for the 905 corpus repositories
  refsample_members.csv          100 random repositories per stratum
  refsample_activity.csv         contributors/commits/releases for the reference sample
  refsample_filtered.csv         the 13 corpus filters re-applied to the reference sample
                                 (outcome == "accepted" is the filtered reference sample)
Inputs (data/01_corpus/):
  repos_pilot_v3.csv             corpus attributes at collection time
  trees_pilot_v3.csv.gz          corpus file trees (tree-entry counts)

Outputs (results/representativeness/):
  table_population.csv           per-stratum population, corpus, share, filter pass rate
  table_representativeness.csv   medians [IQR] and standardised mean differences
  within_corpus_models.txt       logistic models of repository drift on activity attributes

Standardised mean difference (SMD): computed within each stratum on log1p-transformed
counts as (mean_corpus - mean_reference) / pooled SD, then pooled across strata with
the corpus stratum sizes as weights.  Positive values mean the corpus is larger.
"""
import csv, gzip, os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = os.path.join(ROOT, "data", "01_corpus")
R = os.path.join(C, "representativeness")
OUT = os.path.join(ROOT, "results", "representativeness"); os.makedirs(OUT, exist_ok=True)
COLLECTION_DATE = pd.Timestamp("2026-03-23", tz="UTC")   # corpus collection (push windows anchored here)

def band(s): return "10-99" if s < 100 else "100-499" if s < 500 else "500-1999" if s < 2000 else "2000-*"
def stratum(stars, days): return f"stars:{band(stars)}|pushed:{'0-180d' if days <= 180 else '181-730d'}"

# ------------------------------------------------------------------ corpus
corpus = pd.read_csv(os.path.join(C, "repos_pilot_v3.csv"))
corpus["days_since_push"] = (COLLECTION_DATE - pd.to_datetime(corpus.pushed_at)).dt.days
corpus["age_y"] = (COLLECTION_DATE - pd.to_datetime(corpus.created_at)).dt.days / 365.25
corpus["stratum"] = [stratum(s, d) for s, d in zip(corpus.stars, corpus.days_since_push)]
tree_n = {}
with gzip.open(os.path.join(C, "trees_pilot_v3.csv.gz"), "rt") as f:
    rd = csv.reader(f); hdr = next(rd); i = hdr.index("full_name")
    for row in rd: tree_n[row[i]] = tree_n.get(row[i], 0) + 1
corpus["n_tree_entries"] = corpus.full_name.map(tree_n)
act_c = pd.read_csv(os.path.join(R, "corpus905_activity.csv"))
act_c = act_c[act_c.status == 200]
corpus = corpus.merge(act_c[["full_name", "contributors", "commits", "releases", "license"]], on="full_name", how="left")
lic = lambda x: (x.notna() & (x != "NOASSERTION")).astype(float)   # recognised SPDX licence
corpus["license_present"] = lic(corpus.license).where(corpus.full_name.isin(act_c.full_name))   # over repositories whose metadata was retrieved

# ------------------------------------------------------------------ reference sample
members = pd.read_csv(os.path.join(R, "refsample_members.csv"))
act_r = pd.read_csv(os.path.join(R, "refsample_activity.csv"))
act_r = act_r[act_r.status == 200]
filt = pd.read_csv(os.path.join(R, "refsample_filtered.csv"))
FETCH_DATE = pd.to_datetime(act_r.pushed_at).max().normalize()
ref = members.merge(act_r.drop(columns=["status"]), on="full_name", how="left") \
             .merge(filt[["full_name", "outcome", "n_tree_entries", "has_workflow", "has_test_dir",
                          "has_container", "has_makefile"]], on="full_name", how="left")
ref = ref.rename(columns={"stars_now": "stars", "forks_now": "forks", "open_issues": "open_issues"})
ref["days_since_push"] = (FETCH_DATE - pd.to_datetime(ref.pushed_at)).dt.days
ref["age_y"] = (FETCH_DATE - pd.to_datetime(ref.created_at)).dt.days / 365.25
ref["license_present"] = lic(ref.license)
ref_f = ref[ref.outcome == "accepted"].copy()
print(f"reference sample fetched {FETCH_DATE.date()}: {len(ref)} members, {len(ref_f)} pass the filters")

# ------------------------------------------------------------------ Table: population
pop = pd.read_csv(os.path.join(R, "population_stratum_counts.csv"))
order = [f"stars:{b}|pushed:{w}" for b in ["10-99", "100-499", "500-1999", "2000-*"] for w in ["0-180d", "181-730d"]]
t1 = pop.set_index("stratum").loc[order]
t1["corpus"] = corpus.stratum.value_counts().reindex(order).values
t1["corpus_pct"] = (100 * t1.corpus / t1.n_population).round(2)
t1["filters_passed_pct"] = (100 * ref.groupby("stratum").outcome.apply(lambda s: (s == "accepted").mean())).reindex(order).round(0).values
tot = pd.DataFrame({"n_population": [t1.n_population.sum()], "corpus": [t1.corpus.sum()],
                    "corpus_pct": [round(100 * t1.corpus.sum() / t1.n_population.sum(), 2)],
                    "filters_passed_pct": [round(100 * (ref.outcome == "accepted").mean())]}, index=["Total"])
t1 = pd.concat([t1, tot]); t1.to_csv(os.path.join(OUT, "table_population.csv")); print(t1)

# ------------------------------------------------------------------ Table: representativeness
def smd(a, b, col, weights):
    num = den = 0.0
    for s, w in weights.items():
        x = np.log1p(a.loc[a.stratum == s, col].dropna()); y = np.log1p(b.loc[b.stratum == s, col].dropna())
        if len(x) < 2 or len(y) < 2: continue
        sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        if sd > 0: num += w * (x.mean() - y.mean()) / sd; den += w
    return num / den
def med_iqr(v):
    v = v.dropna(); q = v.quantile([.25, .5, .75]).round(0).astype(int)
    return f"{q[.5]} [{q[.25]}, {q[.75]}]"
W = corpus.stratum.value_counts().to_dict()
rows = []
for label, col in [("Stars", "stars"), ("Forks", "forks"), ("Open issues", "open_issues"), ("Age (years)", "age_y"),
                   ("Days since last push", "days_since_push"), ("Repository size (KB)", "size_kb"),
                   ("Tree entries", "n_tree_entries"), ("Contributors", "contributors"), ("Commits", "commits"),
                   ("Releases", "releases")]:
    if col not in corpus and col == "size_kb": corpus[col] = corpus.full_name.map(act_c.set_index("full_name").size_kb)
    rows.append({"attribute": label, "corpus": med_iqr(corpus[col]) if col != "age_y" else
                 f"{corpus[col].median():.1f} [{corpus[col].quantile(.25):.1f}, {corpus[col].quantile(.75):.1f}]",
                 "filtered_ref": med_iqr(ref_f[col]) if col != "age_y" else
                 f"{ref_f[col].median():.1f} [{ref_f[col].quantile(.25):.1f}, {ref_f[col].quantile(.75):.1f}]",
                 "smd_filtered": round(smd(corpus, ref_f, col, W), 2), "smd_unfiltered": round(smd(corpus, ref, col, W), 2)})
ref = ref.merge(filt[["full_name", "has_source_dir", "has_manifest"]], on="full_name", how="left"); ref_f = ref[ref.outcome == "accepted"].copy()
cohen_h = lambda p, q: 2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(q))   # effect size for binary indicators
for label, col in [("Recognised licence (%)", "license_present"), ("Packaging manifest (%)", "has_manifest"),
                   ("Source directory (%)", "has_source_dir"), ("Test directory (%)", "has_test_dir"),
                   ("CI workflow (%)", "has_workflow"), ("Container file (%)", "has_container"), ("Makefile (%)", "has_makefile")]:
    p, q = corpus[col].mean(), ref_f[col].mean()
    rows.append({"attribute": label, "corpus": round(100 * p), "filtered_ref": round(100 * q),
                 "smd_filtered": round(cohen_h(p, q), 2), "smd_unfiltered": ""})
t2 = pd.DataFrame(rows); t2.to_csv(os.path.join(OUT, "table_representativeness.csv"), index=False); print(t2.to_string())

# ------------------------------------------------------------------ within-corpus association of drift with activity
try:
    import statsmodels.api as sm
    m = pd.read_csv(os.path.join(ROOT, "data", "07_models", "repo_modeling_dataset_v6.csv"))
    m = m.merge(corpus[["full_name", "contributors", "commits", "releases", "size_kb", "n_tree_entries"]], on="full_name", how="left")
    # each activity attribute is added in turn to the repository-level logistic model of
    # Section 4.7 (log stars, age, days since push, log README words, log claims)
    base = ["log_stars", "age_y", "days_since_push", "log_words", "log_claims"]
    lines = ["Logistic model of any drifted claim per repository (Section 4.7 covariates + one activity term)"]
    for col in ["contributors", "commits", "releases", "size_kb", "n_tree_entries"]:
        d = m.dropna(subset=[col]).copy(); d["x"] = np.log1p(d[col])
        fit = sm.Logit(d["any"], sm.add_constant(d[base + ["x"]])).fit(disp=0)
        lines.append(f"{col:16s} adjusted OR per log unit {np.exp(fit.params['x']):.2f}  p={fit.pvalues['x']:.2f}  n={len(d)}")
    d = m.dropna(subset=["contributors"]).copy(); d["tertile"] = pd.qcut(d.contributors, 3, labels=["low", "mid", "high"])
    rate = d.groupby("tertile", observed=True).apply(lambda g: 100 * g.drifted.sum() / g.n.sum())
    lines.append("claim-level drift rate by contributor tertile (%): " + ", ".join(f"{k}={v:.1f}" for k, v in rate.items()))
    open(os.path.join(OUT, "within_corpus_models.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
except ImportError:
    print("statsmodels not installed; within-corpus models skipped")

# ------------------------------------------------------------------ prevalence reweighted to population shares (Section 4.4)
m = pd.read_csv(os.path.join(ROOT, "data", "07_models", "repo_modeling_dataset_v6.csv"))
m = m.merge(corpus[["full_name", "stratum"]], on="full_name")
g = m.groupby("stratum").agg(repos=("n", "size"), claims=("n", "sum"), drifted=("drifted", "sum"), any_drift=("any", "sum"))
g["pop"] = pop.set_index("stratum").n_population
m["wt"] = m.stratum.map(g["pop"] / g["repos"])          # population repositories per sampled repository
def est(d): return (100 * (d.wt * d.drifted).sum() / (d.wt * d.n).sum(), 100 * (d.wt * d["any"]).sum() / d.wt.sum())
rng = np.random.default_rng(1); idx = {s: m.index[m.stratum == s].values for s in g.index}; boot = []
for _ in range(2000):
    boot.append(est(m.loc[np.concatenate([rng.choice(idx[s], len(idx[s])) for s in g.index])]))
boot = np.array(boot); point = est(m)
lines = [f"unweighted: claims {100 * m.drifted.sum() / m.n.sum():.1f}%  repositories {100 * m['any'].mean():.1f}%",
         f"population-weighted: claims {point[0]:.1f}% (95% CI {np.percentile(boot[:, 0], 2.5):.1f}-{np.percentile(boot[:, 0], 97.5):.1f}), "
         f"repositories {point[1]:.1f}% (95% CI {np.percentile(boot[:, 1], 2.5):.1f}-{np.percentile(boot[:, 1], 97.5):.1f})",
         "per-stratum claim rate (%): " + ", ".join(f"{s}={100 * r.drifted / r.claims:.1f}" for s, r in g.iterrows()),
         "per-stratum repository rate (%): " + ", ".join(f"{s}={100 * r.any_drift / r.repos:.1f}" for s, r in g.iterrows())]
open(os.path.join(OUT, "weighted_prevalence.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
