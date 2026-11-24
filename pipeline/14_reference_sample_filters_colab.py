# ============================================================
# 14_reference_sample_filters.py  (Google Colab)
# Applies the thirteen corpus filters of 01_corpus_sampling.py, unchanged, to the
# random reference sample of the stratum population (refsample_members.csv from
# Script 13), so that the corpus can be compared with a filtered random sample
# of the same population (Section 3.3 of the article). For each reference repository it fetches
# the metadata, README and default-branch tree, runs the filters in the same
# order, and records the outcome and the same structural indicators the corpus
# file carries. Checkpointed; re-run to resume. About 2,400 API calls (~45 min).
# OUTPUT: My Drive/README_Drift/representativeness/refsample_filtered.csv
# ============================================================
TOKEN = ""
DRIVE_DIR = "README_Drift/representativeness"

import os, re, time, base64, json, datetime as dt
import requests, pandas as pd
from datetime import datetime, timedelta, timezone
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
OUT = f"/content/drive/My Drive/{DRIVE_DIR}"
if not TOKEN:
    from getpass import getpass
    TOKEN = getpass("GitHub token: ").strip()
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "readme-drift-representativeness"}
_chk = requests.get("https://api.github.com/rate_limit", headers=H, timeout=30)
if _chk.status_code != 200:
    raise SystemExit(f"Token rejected by GitHub (HTTP {_chk.status_code}). Paste a valid token into TOKEN = \"...\" and run again.")
print("token OK; core requests remaining this hour:", _chk.json()["resources"]["core"]["remaining"])
LAST_STATUS = {"code": None}

def github_get(url, params=None, retries=6):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=H, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(3 + attempt * 3); continue
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset"); retry = r.headers.get("Retry-After")
            wait = int(retry) if retry and retry.isdigit() else (max(5, int(reset) - int(time.time()) + 2) if reset else 60)
            print(f"rate limited; sleeping {wait}s"); time.sleep(min(wait, 1800)); continue
        if r.status_code >= 500:
            time.sleep(5 + attempt * 5); continue
        LAST_STATUS["code"] = r.status_code
        if r.status_code >= 400:
            return None
        return r
    return None

def get_readme(owner, repo):
    r = github_get(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if r is None: return None, None
    data = r.json(); content = data.get("content"); path = data.get("path")
    if content and data.get("encoding") == "base64":
        try: return base64.b64decode(content).decode("utf-8", errors="replace"), path
        except Exception: return None, path
    return None, path

def get_recursive_tree(owner, repo, branch):
    r = github_get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": 1})
    return r.json().get("tree", []) if r is not None else []

# ---- constants and filter functions copied verbatim from 01_corpus_sampling.py ----

# Hard scope controls
MIN_README_WORDS = 40
MIN_ALPHA_TOKENS_FOR_ENGLISH = 25
MIN_RECENT_PUSH_DAYS = 730
MAX_CANDIDATES_PER_STRATUM = 250


WHITELIST_CONFIG_BASENAMES = {
    "package.json",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "makefile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "cargo.toml",
    "go.mod",
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pipfile",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}

SOURCE_DIR_HINTS = {"src", "app", "lib", "cmd", "server", "client", "package", "packages"}
TEST_DIR_HINTS = {"tests", "test", "spec", "specs"}
DOC_DIR_HINTS = {"docs", "doc", "images", "assets", ".github"}
MANIFEST_BASENAMES = {
    "package.json", "pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "makefile",
    "pom.xml", "build.gradle", "build.gradle.kts", "cargo.toml", "go.mod", "requirements.txt",
    "environment.yml", "environment.yaml", "pipfile", "dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "compose.yml", "compose.yaml"
}
PYTHON_FILE_RE = re.compile(r"\.py$", re.I)
NOTEBOOK_RE = re.compile(r"\.ipynb$", re.I)

EXCLUDE_NAME_PATTERNS = [
    r"\bawesome\b",
    r"\bprimer\b",
    r"\bbooks?\b",
    r"\bresources?\b",
    r"\btutorial\b",
    r"\binterview\b",
    r"\broadmap\b",
    r"\bpublic[- ]?apis\b",
    r"\bcurated\b",
    r"\blist\b",
    r"\bcheatsheet\b",
    r"\bcheat[- ]?sheet\b",
    r"\bboilerplate\b",
    r"\btemplate\b",
]

EXCLUDE_DESC_PATTERNS = [
    r"\ba curated list\b",
    r"\bcollection of\b",
    r"\blist of\b",
    r"\bfree programming books\b",
    r"\binterview questions\b",
    r"\blearning resource\b",
    r"\bawesome\b",
    r"\broadmap\b",
    r"\btutorial\b",
    r"\bguide\b",
    r"\bboilerplate\b",
    r"\btemplate\b",
    r"\bstarter\b",
]

CURATED_README_PATTERNS = [
    r"(?im)^\s*[-*+]\s*\[[^\]]+\]\([^\)]+\)",
    r"(?im)^\s*\d+\.\s*\[[^\]]+\]\([^\)]+\)",
    r"(?i)curated list",
    r"(?i)awesome list",
    r"(?i)collection of resources",
    r"(?i)learning resources",
    r"(?i)free .* books",
]

COMMON_ENGLISH_FUNCTION_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "you", "your", "are", "is", "to",
    "of", "in", "on", "as", "by", "or", "an", "be", "can", "will", "use", "using", "install",
    "run", "build", "test", "project", "application", "example", "examples", "quick", "start",
    "getting", "started", "requirements", "setup", "deployment", "development", "usage", "command",
    "commands", "python", "repository", "clone", "configure", "docker", "service", "server", "client",
    "how", "installing", "running", "testing", "module", "package", "library", "framework", "api"
}


# -----------------------------
# API helpers
# -----------------------------

# -----------------------------

def strip_markdown_for_language_signal(text):
    t = str(text or "")
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"`[^`\n]+`", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", t)
    t = re.sub(r"\[[^\]]+\]\([^\)]+\)", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[#>*_\-~=|]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def readme_word_count(text):
    return len(re.findall(r"\b\w+\b", str(text or "")))


def english_readme_ok(text):
    prose = strip_markdown_for_language_signal(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", prose)
    alpha_tokens = [t.lower() for t in tokens if re.search(r"[A-Za-z]", t)]

    if len(alpha_tokens) < MIN_ALPHA_TOKENS_FOR_ENGLISH:
        return False, "insufficient_natural_language_for_english_check"

    ascii_chars = sum(1 for ch in prose if ord(ch) < 128)
    ascii_ratio = ascii_chars / max(1, len(prose))
    if ascii_ratio < 0.85:
        return False, "non_english_readme_low_ascii_ratio"

    function_hits = sum(1 for t in alpha_tokens if t in COMMON_ENGLISH_FUNCTION_WORDS)
    function_ratio = function_hits / max(1, len(alpha_tokens))
    if function_ratio < 0.06:
        return False, "weak_english_function_word_signal"

    alpha_ratio = len(alpha_tokens) / max(1, len(re.findall(r"\S+", prose)))
    if alpha_ratio < 0.45:
        return False, "readme_too_code_or_markup_heavy"

    return True, "english_ok"


def repo_name_or_desc_excluded(full_name, description):
    hay = f"{full_name} {description or ''}".lower()
    for pat in EXCLUDE_NAME_PATTERNS + EXCLUDE_DESC_PATTERNS:
        if re.search(pat, hay, flags=re.I):
            return True
    return False


def looks_like_curated_list(readme_text):
    t = str(readme_text or "")
    if any(re.search(p, t) for p in CURATED_README_PATTERNS):
        bullet_link_lines = len(re.findall(r"(?im)^\s*[-*+]\s*\[[^\]]+\]\([^\)]+\)", t))
        numbered_link_lines = len(re.findall(r"(?im)^\s*\d+\.\s*\[[^\]]+\]\([^\)]+\)", t))
        if bullet_link_lines + numbered_link_lines >= 8:
            return True
    return False

# -----------------------------
# Tree summarization / screening
# -----------------------------

def summarize_tree(tree):
    file_paths = []
    dir_paths = []
    basenames = []
    top_dirs = set()

    for node in tree:
        p = str(node.get("path") or "")
        t = str(node.get("type") or "")
        if not p:
            continue
        lower_p = p.lower().replace("\\", "/")
        if t == "blob":
            file_paths.append(lower_p)
            basenames.append(os.path.basename(lower_p))
            if "/" in lower_p:
                top_dirs.add(lower_p.split("/")[0])
        elif t == "tree":
            dir_paths.append(lower_p)
            top_dirs.add(lower_p.split("/")[0] if "/" in lower_p else lower_p)

    basenames_set = set(basenames)
    top_dirs_set = set(top_dirs)
    dir_paths_set = set(dir_paths)
    file_paths_set = set(file_paths)

    has_manifest = int(any(b in MANIFEST_BASENAMES for b in basenames_set))
    has_container = int(any(b in {"dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"} for b in basenames_set))
    has_makefile = int("makefile" in basenames_set)
    has_workflow = int(any(p.startswith(".github/workflows/") for p in file_paths_set))
    has_source_dir = int(any(d in top_dirs_set or d in dir_paths_set for d in SOURCE_DIR_HINTS))
    has_test_dir = int(any(d in top_dirs_set or d in dir_paths_set for d in TEST_DIR_HINTS))
    doc_dir_count = sum(1 for d in DOC_DIR_HINTS if d in top_dirs_set or d in dir_paths_set)

    py_files = [p for p in file_paths_set if PYTHON_FILE_RE.search(p)]
    notebook_files = [p for p in file_paths_set if NOTEBOOK_RE.search(p)]
    has_python_files = int(len(py_files) > 0)
    has_notebook_only = int(len(notebook_files) > 0 and len(py_files) == 0)

    return {
        "file_paths": file_paths_set,
        "dir_paths": dir_paths_set,
        "top_dirs": top_dirs_set,
        "basenames": basenames_set,
        "file_count": len(file_paths_set),
        "dir_count": len(dir_paths_set),
        "py_file_count": len(py_files),
        "notebook_count": len(notebook_files),
        "has_manifest": has_manifest,
        "has_container": has_container,
        "has_makefile": has_makefile,
        "has_workflow": has_workflow,
        "has_source_dir": has_source_dir,
        "has_test_dir": has_test_dir,
        "has_python_files": has_python_files,
        "has_notebook_only": has_notebook_only,
        "doc_dir_count": doc_dir_count,
    }


def repo_looks_like_software_project(full_name, description, readme_text, tree_info, pushed_at):
    desc = str(description or "")
    name_desc = f"{full_name} {desc}".lower()

    if pushed_at:
        try:
            pushed_dt = datetime.fromisoformat(str(pushed_at).replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - pushed_dt).days
            if age_days > MIN_RECENT_PUSH_DAYS:
                return False, "inactive_repo"
        except Exception:
            pass

    if repo_name_or_desc_excluded(full_name, description):
        return False, "excluded_by_name_or_description"

    if readme_word_count(readme_text) < MIN_README_WORDS:
        return False, "readme_too_short"

    if looks_like_curated_list(readme_text):
        return False, "curated_list_heuristic"

    if re.search(r"\b(dataset|benchmark data|paper list|reading list|syllabus|course notes)\b", name_desc, flags=re.I):
        return False, "non_project_repo_semantics"

    if tree_info["has_notebook_only"]:
        return False, "notebook_only_repo"

    if not tree_info["has_python_files"]:
        return False, "no_python_files"

    # Transparent structural score to operationalize "software project".
    structural_score = (
        tree_info["has_source_dir"] +
        tree_info["has_test_dir"] +
        tree_info["has_manifest"] +
        tree_info["has_workflow"] +
        tree_info["has_container"] +
        tree_info["has_makefile"]
    )

    if tree_info["py_file_count"] < 3 and structural_score < 2:
        return False, "insufficient_substantive_code"

    if structural_score < 2:
        return False, "insufficient_software_structure"

    if tree_info["doc_dir_count"] >= 2 and structural_score < 3:
        return False, "doc_heavy_structure"

    return True, "accepted"

# -----------------------------
# Candidate collection
# -----------------------------


# ---- apply the filters to the reference sample, in the original order ----
members = pd.read_csv(f"{OUT}/refsample_members.csv")
path = f"{OUT}/refsample_filtered.csv"
prev = None
if os.path.exists(path):
    try:
        prev = pd.read_csv(path, engine="python", on_bad_lines="skip")
        prev = prev[prev.outcome != "repo_unavailable"]      # redo rows from a failed run
    except Exception:
        prev = None
    os.remove(path)
    if prev is not None and len(prev):
        prev.to_csv(path, index=False)
done = set(prev.full_name) if prev is not None and len(prev) else set()
todo = [t for t in members.itertuples(index=False) if t.full_name not in done]
print(f"{len(todo)} to process ({len(done)} done)")
buf = []
def flush():
    global buf
    if buf:
        pd.DataFrame(buf).to_csv(path, mode="a", index=False, header=not os.path.exists(path)); buf = []
for i, t in enumerate(todo, 1):
    full_name = t.full_name; owner, repo = full_name.split("/", 1)
    row = {"stratum": t.stratum, "full_name": full_name, "outcome": None, "readme_word_count": None,
           "has_manifest": None, "has_container": None, "has_makefile": None, "has_workflow": None,
           "has_source_dir": None, "has_test_dir": None, "py_file_count": None, "n_tree_entries": None}
    r = github_get(f"https://api.github.com/repos/{full_name}")
    if r is None:
        row["outcome"] = "repo_unavailable"; row["http_status"] = LAST_STATUS["code"]; buf.append(row)
    else:
        j = r.json(); description = j.get("description"); pushed_at = j.get("pushed_at"); branch = j.get("default_branch") or "main"
        row.update({"stars_now": j.get("stargazers_count"), "forks_now": j.get("forks_count"), "size_kb": j.get("size"),
                    "created_at": j.get("created_at"), "pushed_at": pushed_at, "open_issues": j.get("open_issues_count"),
                    "license": (j.get("license") or {}).get("spdx_id"), "archived": j.get("archived"), "fork": j.get("fork")})
        if repo_name_or_desc_excluded(full_name, description):
            row["outcome"] = "excluded_by_name_or_description"
        else:
            readme_text, readme_path = get_readme(owner, repo)
            if not readme_text:
                row["outcome"] = "missing_or_unreadable_readme"
            else:
                row["readme_word_count"] = readme_word_count(readme_text)
                ok, reason = english_readme_ok(readme_text)
                if not ok:
                    row["outcome"] = reason
                elif looks_like_curated_list(readme_text):
                    row["outcome"] = "curated_list_heuristic"
                else:
                    tree = get_recursive_tree(owner, repo, branch)
                    if not tree and branch != "master":
                        tree = get_recursive_tree(owner, repo, "master")
                    if not tree:
                        row["outcome"] = "tree_unavailable"
                    else:
                        info = summarize_tree(tree)
                        ok2, reason2 = repo_looks_like_software_project(full_name, description, readme_text, info, pushed_at)
                        row["outcome"] = reason2
                        row.update({k: info.get(k) for k in ("has_manifest", "has_container", "has_makefile", "has_workflow",
                                                              "has_source_dir", "has_test_dir", "py_file_count")})
                        row["n_tree_entries"] = len(tree)
        buf.append(row)
    if i % 20 == 0 or i == len(todo):
        flush(); print(f"{i}/{len(todo)}  last: {full_name} -> {row['outcome']}")
flush()
res = pd.read_csv(path)
print(res.groupby("stratum").outcome.apply(lambda s: (s == "accepted").mean()).rename("acceptance_rate"))
print(res.outcome.value_counts())
print("done")
