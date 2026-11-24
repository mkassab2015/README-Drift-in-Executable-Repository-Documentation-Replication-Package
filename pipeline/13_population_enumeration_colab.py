# ============================================================
# 13_representativeness.py  (Google Colab)
# Representativeness check of the README Drift corpus (Section 3.3 of the article).
#
# Part A  Enumerates the FULL population of each of the eight sampling strata
#         (language:Python fork:false archived:false, 4 star bands x 2 push windows)
#         through the GitHub search API, sharding the `created:` range until every
#         query returns < 1,000 results, so that the 1,000-result cap is bypassed.
#         Saves every repository with the attributes the search API returns
#         (stars, forks, size, created_at, pushed_at, open_issues, license, topics).
# Part B  Fetches contributor count, commit count, size and license for the 905
#         corpus repositories and for a random reference sample of 100 repositories
#         per stratum drawn from the enumerated population.
#
# Everything is checkpointed to Drive, so the cell can be re-run after a timeout
# and will continue where it stopped.
#
# HOW TO RUN: Runtime > Run all. When asked, authorise Drive. Paste a GitHub
# personal access token (classic, no scopes needed; "public_repo" is enough)
# into the TOKEN box below or in the prompt. Expected time: 45 to 90 minutes.
# OUTPUT: My Drive/README_Drift/representativeness/
#   population_all.csv.gz           every repository in the eight strata (Part A)
#   population_stratum_counts.csv   number of repositories per stratum
#   corpus905_activity.csv          contributors, commits, size, license for the corpus (Part B)
#   refsample_activity.csv          same for the reference sample (Part B)
#   run_log.txt
# ============================================================

TOKEN = ""            # <- paste token here, or leave empty to be prompted
DRIVE_DIR = "README_Drift/representativeness"
CORPUS_CSV = "README_Drift/repos_pilot_v3.csv"   # path in My Drive to the corpus file (905 rows)
REF_PER_STRATUM = 100
SEED = 20260905

import os, sys, time, json, math, random, datetime as dt
import requests, pandas as pd

from google.colab import drive
drive.mount("/content/drive", force_remount=False)
OUT = f"/content/drive/My Drive/{DRIVE_DIR}"
os.makedirs(OUT, exist_ok=True)
if not TOKEN:
    from getpass import getpass
    TOKEN = getpass("GitHub token: ").strip()
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "readme-drift-representativeness"}
LOG = open(f"{OUT}/run_log.txt", "a")
def log(msg):
    s = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  {msg}"
    print(s); LOG.write(s + "\n"); LOG.flush()

# ---------------------------------------------------------------- HTTP with rate-limit handling
def get(url, params=None, kind="core"):
    for attempt in range(8):
        r = requests.get(url, headers=H, params=params, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset"); retry = r.headers.get("Retry-After")
            wait = int(retry) if retry else (max(5, int(reset) - int(time.time()) + 2) if reset else 60)
            log(f"rate limited ({kind}); sleeping {wait}s"); time.sleep(min(wait, 900)); continue
        if r.status_code in (502, 503, 504):
            time.sleep(10 * (attempt + 1)); continue
        if r.status_code in (404, 409, 451):
            return r
        log(f"HTTP {r.status_code} for {url} {params}: {r.text[:120]}"); time.sleep(5)
    return r

def search_count(q):
    r = get("https://api.github.com/search/repositories", {"q": q, "per_page": 1}, "search")
    return r.json().get("total_count", 0) if r.status_code == 200 else None

# ---------------------------------------------------------------- strata (same definitions as 01_corpus_sampling.py)
TODAY = dt.date.today()
def d(n): return (TODAY - dt.timedelta(days=n)).isoformat()
STAR_BANDS = [(10, 99), (100, 499), (500, 1999), (2000, "*")]
PUSH_WINDOWS = [("0-180d", d(180), TODAY.isoformat()), ("181-730d", d(730), d(181))]
STRATA = []
for smin, smax in STAR_BANDS:
    for wname, p0, p1 in PUSH_WINDOWS:
        STRATA.append({"stratum": f"stars:{smin}-{smax}|pushed:{wname}",
                       "q": f"language:Python fork:false archived:false stars:{smin}..{smax} pushed:{p0}..{p1}"})
log(f"enumeration date {TODAY}; strata anchored to this date (push windows 0-180 and 181-730 days before it)")

# ---------------------------------------------------------------- Part A: sharded enumeration
FIELDS = ["full_name", "stargazers_count", "forks_count", "size", "created_at", "pushed_at", "updated_at",
          "open_issues_count", "watchers_count", "has_issues", "has_wiki", "has_pages", "has_discussions",
          "language", "default_branch", "topics", "license"]
def rows_from(items, stratum):
    out = []
    for it in items:
        row = {k: it.get(k) for k in FIELDS}
        row["license"] = (it.get("license") or {}).get("spdx_id")
        row["topics"] = "|".join(it.get("topics") or [])
        row["stratum"] = stratum
        out.append(row)
    return out

def enumerate_range(q_base, c0, c1, stratum, part_path, done_shards):
    """Enumerate all repositories created in [c0, c1] for query q_base; bisect while > 1000.
    Each leaf shard is appended to part_path as soon as it is fetched (checkpoint)."""
    key = f"{c0}..{c1}"
    if key in done_shards: return
    q = f"{q_base} created:{c0.isoformat()}..{c1.isoformat()}"
    n = search_count(q)
    if n is None: log(f"count failed for {q}"); return
    if n == 0:
        done_shards.add(key); return
    if n > 1000 and c0 < c1:
        mid = c0 + (c1 - c0) // 2
        enumerate_range(q_base, c0, mid, stratum, part_path, done_shards)
        enumerate_range(q_base, mid + dt.timedelta(days=1), c1, stratum, part_path, done_shards)
        return
    if n > 1000:
        log(f"single-day range {c0} has {n} > 1000; only the first 1,000 are retrievable")
    sink = []
    for page in range(1, math.ceil(min(n, 1000) / 100) + 1):
        r = get("https://api.github.com/search/repositories",
                {"q": q, "per_page": 100, "page": page, "sort": "stars", "order": "desc"}, "search")
        if r.status_code != 200: log(f"page {page} failed for {q}"); break
        items = r.json().get("items", [])
        sink.extend(rows_from(items, stratum))
        if len(items) < 100: break
    if sink:
        pd.DataFrame(sink).to_csv(part_path, mode="a", index=False, header=not os.path.exists(part_path))
    done_shards.add(key)
    log(f"  shard created:{key}: {n} repositories ({len(sink)} fetched)")

POP = f"{OUT}/population_all.csv.gz"; DONE = f"{OUT}/population_done_strata.json"
done = json.load(open(DONE)) if os.path.exists(DONE) else []
for s_ in STRATA:
    if s_["stratum"] in done: log(f"skip {s_['stratum']} (done)"); continue
    log(f"enumerating {s_['stratum']}: {s_['q']}")
    safe = s_["stratum"].replace(":", "_").replace("|", "_")
    part = f"{OUT}/part_{safe}.csv"; shards_file = f"{OUT}/part_{safe}_shards.json"
    done_shards = set(json.load(open(shards_file))) if os.path.exists(shards_file) else set()
    try:
        enumerate_range(s_["q"], dt.date(2008, 1, 1), TODAY, s_["stratum"], part, done_shards)
    finally:
        json.dump(sorted(done_shards), open(shards_file, "w"))
    df = pd.read_csv(part).drop_duplicates("full_name") if os.path.exists(part) else pd.DataFrame(columns=FIELDS + ["stratum"])
    df.to_csv(POP, mode="a", index=False, header=not os.path.exists(POP), compression="gzip")
    done.append(s_["stratum"]); json.dump(done, open(DONE, "w"))
    log(f"  {s_['stratum']}: {len(df)} repositories")
pop = pd.read_csv(POP).drop_duplicates(["stratum", "full_name"])
pop.groupby("stratum").size().rename("n_population").to_csv(f"{OUT}/population_stratum_counts.csv")
log(f"population total {len(pop)}: " + ", ".join(f"{k}={v}" for k, v in pop.groupby('stratum').size().items()))

# ---------------------------------------------------------------- Part B: activity attributes
def last_page_count(url, params):
    """Number of items via the Link header's last page (per_page=1)."""
    r = get(url, dict(params, per_page=1), "core")
    if r.status_code != 200: return None
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        m = re.search(r'[?&]page=(\d+)>; rel="last"', link)
        if m: return int(m.group(1))
    return len(r.json()) if isinstance(r.json(), list) else None

def activity(full_name):
    base = f"https://api.github.com/repos/{full_name}"
    r = get(base)
    if r.status_code != 200:
        return {"full_name": full_name, "status": r.status_code}
    j = r.json()
    return {"full_name": full_name, "status": 200,
            "stars_now": j.get("stargazers_count"), "forks_now": j.get("forks_count"), "size_kb": j.get("size"),
            "created_at": j.get("created_at"), "pushed_at": j.get("pushed_at"), "open_issues": j.get("open_issues_count"),
            "license": (j.get("license") or {}).get("spdx_id"), "archived": j.get("archived"),
            "subscribers": j.get("subscribers_count"), "network": j.get("network_count"),
            "contributors": last_page_count(base + "/contributors", {"anonymous": "true"}),
            "commits": last_page_count(base + "/commits", {}),
            "releases": last_page_count(base + "/releases", {})}

def run_activity(names, path, label):
    have = set(pd.read_csv(path).full_name) if os.path.exists(path) else set()
    todo = [n for n in names if n not in have]
    log(f"{label}: {len(todo)} repositories to fetch ({len(have)} already done)")
    buf = []
    for i, n in enumerate(todo, 1):
        buf.append(activity(n))
        if i % 25 == 0 or i == len(todo):
            pd.DataFrame(buf).to_csv(path, mode="a", index=False, header=not os.path.exists(path)); buf = []
            log(f"  {label}: {i}/{len(todo)}")

corpus = pd.read_csv(f"/content/drive/My Drive/{CORPUS_CSV}")
run_activity(corpus.full_name.tolist(), f"{OUT}/corpus905_activity.csv", "corpus")

rng = random.Random(SEED)
ref = []
for s, g in pop.groupby("stratum"):
    names = sorted(g.full_name.tolist()); rng.shuffle(names)
    ref += [(s, n) for n in names[:REF_PER_STRATUM]]
pd.DataFrame(ref, columns=["stratum", "full_name"]).to_csv(f"{OUT}/refsample_members.csv", index=False)
run_activity([n for _, n in ref], f"{OUT}/refsample_activity.csv", "reference sample")
log("done")
