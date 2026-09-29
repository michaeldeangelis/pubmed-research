"""Outcome C: field influence (iCite RCR) and a disruption score.

Run as ``python -m src.meta.impact``. Writes ``data/meta/impact.jsonl`` (one
row per PMID in ``data/meta/papers.jsonl``) and
``data/meta/manifest_impact.json``.

Columns (spec): pmid, rcr, cd_source, cd, n_refs_openalex. Extra columns
that document the score: n_refs_icite, n_citers_5y, n_i, n_j, openalex_id,
cd_openalex_nok (validation subsample only), sciscinet_d (auxiliary, see
below).

Disruption source, decided on 2026-09-29 (details in the manifest):

* SciSciNet CD5 is not usable. SciSciNet v2 (the release that carries CD5)
  is a gated Hugging Face dataset (HTTP 401 without an accepted licence and
  a login). SciSciNet v1 is open, but its ``Disruption`` column is the Wu et
  al. (2019) D over all citations up to the MAG 2021 snapshot, not CD5.
  It is joined by PMID (``df_magid_pubmedid.tsv``) or DOI and written as the
  auxiliary column ``sciscinet_d`` when ``--sciscinet`` is passed; it is
  never written to ``cd``.
* The OpenAlex route is not feasible at corpus scale. Anonymous OpenAlex
  calls are limited to 1000 credits per day (one credit per list request),
  and the 5-year citer lists for ~12k papers need well over 12k requests.
* So ``cd`` is the n_k-free variant ``(n_i - n_j) / (n_i + n_j)`` computed
  on the NIH Open Citation Collection (iCite): focal references and dated
  citers come from ``data/meta/icite.jsonl``; each citer's reference list
  is fetched from iCite. ``cd_source = "icite_nok"``. iCite covers PubMed
  items only, so references and citers outside PubMed are not seen.
* The same variant is computed from OpenAlex (``cd_openalex_nok``) for a
  random validation subsample (research articles, rng seed 0, processed in
  a fixed order until the daily OpenAlex budget runs out), so the two
  graph sources can be compared. OpenAlex also gives ``n_refs_openalex``
  for every paper (batched lookups, 50 PMIDs per request).

Definition: with Y the paper's publication year (``papers.jsonl``), the
5-year citers are citing PMIDs with Y <= citing year <= Y + 5. n_j counts
citers that also cite at least one of the focal paper's references; n_i
counts the rest. ``cd`` is null when the focal paper has no references in
the source, or no 5-year citers.

Everything fetched is cached under ``data/meta/cache/`` (``icite/``,
``openalex/``, ``sciscinet/``), so reruns are cheap and interrupted runs
resume.

Blinding: this module never reads ``outcome_b.jsonl`` or
``features.jsonl`` and relates nothing to anything; the manifest holds only
marginal counts of this table.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import requests

from src import config

logger = logging.getLogger(__name__)

META_DIR = config.DATA / "meta"
CACHE_DIR = META_DIR / "cache"
PAPERS_PATH = META_DIR / "papers.jsonl"
ICITE_PATH = META_DIR / "icite.jsonl"
OUT_PATH = META_DIR / "impact.jsonl"
MANIFEST_PATH = META_DIR / "manifest_impact.json"

ICITE_URL = "https://icite.od.nih.gov/api/pubs"
OPENALEX_URL = "https://api.openalex.org/works"
SCISCINET_BASE = "https://huggingface.co/datasets/Northwestern-CSSI/sciscinet-v1/resolve/main"

WINDOW = 5
ICITE_BATCH = 200  # 500 returns HTTP 413
OPENALEX_BATCH = 50
OPENALEX_CITER_CAP = 2000
OPENALEX_RESERVE = 20  # credits left untouched at the end of a run
VALIDATION_MAX = 300
HEADERS = {"User-Agent": "ras-mapk-metascience-v1"}
SEED = 0


# ---------------------------------------------------------------- pure core


def window_citers(cited_by_by_year: Iterable[dict] | None, year: int | None,
                  window: int = WINDOW) -> tuple[list[str], int]:
    """Citing PMIDs dated within [year, year + window].

    ``cited_by_by_year`` is iCite's ``citedByPmidsByYear`` (a list of
    one-entry dicts {pmid: year}). Returns (citers, n_undated).
    """
    if year is None:
        return [], 0
    out: list[str] = []
    undated = 0
    for entry in cited_by_by_year or []:
        for pmid, cy in entry.items():
            if cy is None:
                undated += 1
                continue
            if year <= int(cy) <= year + window:
                out.append(str(pmid))
    return sorted(set(out)), undated


def disruption_nok(focal_refs: set, citer_refs: dict) -> dict:
    """n_k-free disruption over the given citers.

    ``citer_refs`` maps each citer to the set of works it references (only
    membership in ``focal_refs`` matters). Returns dict with cd, n_i, n_j;
    cd is None when there are no focal references or no citers.
    """
    if not focal_refs or not citer_refs:
        return {"cd": None, "n_i": None if not focal_refs else 0,
                "n_j": None if not focal_refs else 0}
    n_j = sum(1 for refs in citer_refs.values() if refs & focal_refs)
    n_i = len(citer_refs) - n_j
    return {"cd": (n_i - n_j) / (n_i + n_j), "n_i": n_i, "n_j": n_j}


def pick_openalex_work(works: list[dict]) -> dict | None:
    """Several OpenAlex works can carry one PMID; keep the one with most
    references, ties broken by the smaller id (deterministic)."""
    if not works:
        return None
    return sorted(works, key=lambda w: (-len(w.get("referenced_works") or []),
                                        w.get("id") or ""))[0]


def pmid_from_openalex(work: dict) -> str | None:
    pm = (work.get("ids") or {}).get("pmid")
    if not pm:
        return None
    return pm.rstrip("/").rsplit("/", 1)[-1]


def short_id(openalex_url: str) -> str:
    return openalex_url.rstrip("/").rsplit("/", 1)[-1]


def validation_order(pmids: Iterable[str], seed: int = SEED) -> list[str]:
    """Fixed random order for the OpenAlex validation subsample."""
    order = sorted(set(pmids), key=int)
    random.Random(seed).shuffle(order)
    return order


# ---------------------------------------------------------------- IO helpers


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(path)


class JsonlCache:
    """Append-only {key: record} cache; the last record for a key wins."""

    def __init__(self, path: Path, key: str = "pmid"):
        self.path = path
        self.key = key
        self.data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # torn final line
                        continue
                    self.data[str(rec[key])] = rec

    def __contains__(self, k) -> bool:
        return str(k) in self.data

    def get(self, k):
        return self.data.get(str(k))

    def add_many(self, recs: list[dict]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                for rec in recs:
                    self.data[str(rec[self.key])] = rec
                    fh.write(json.dumps(rec) + "\n")


def http_get(url: str, params: dict, tries: int = 4, sleep: float = 0.15,
             session: requests.Session | None = None) -> requests.Response:
    last: Exception | None = None
    get = (session or requests).get
    for attempt in range(tries):
        time.sleep(sleep)
        try:
            resp = get(url, params=params, headers=HEADERS, timeout=60)
            if resp.status_code == 429:
                return resp  # caller decides (budget)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


# ---------------------------------------------------------------- iCite


IciteFetch = Callable[[list[str]], list[dict]]


def icite_fetch(pmids: list[str]) -> list[dict]:
    resp = http_get(ICITE_URL, {"pmids": ",".join(pmids),
                                "fl": "pmid,year,references"})
    return resp.json().get("data", [])


def fetch_citer_refs(pmids: list[str], cache: JsonlCache,
                     fetch: IciteFetch = icite_fetch, workers: int = 4,
                     batch: int = ICITE_BATCH) -> None:
    """Make sure every PMID has an iCite reference record in ``cache``.

    PMIDs iCite does not return are cached as {"pmid", "missing": true}.
    """
    todo = [p for p in dict.fromkeys(str(p) for p in pmids) if p not in cache]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    logger.info("iCite citer refs: %d cached, %d to fetch in %d batches",
                len(set(pmids)) - len(todo), len(todo), len(batches))
    done = [0]

    def run(b: list[str]) -> None:
        got = fetch(b)
        recs = []
        seen = set()
        for r in got:
            p = str(r.get("pmid"))
            seen.add(p)
            recs.append({"pmid": p, "year": r.get("year"),
                         "references": [str(x) for x in r.get("references") or []]})
        recs += [{"pmid": p, "missing": True} for p in b if p not in seen]
        cache.add_many(recs)
        done[0] += 1
        if done[0] % 100 == 0 or done[0] == len(batches):
            logger.info("iCite batches %d/%d", done[0], len(batches))

    if workers <= 1:
        for b in batches:
            run(b)
    else:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(run, batches))


def compute_icite_cd(papers: list[dict], icite: dict[str, dict],
                     cache: JsonlCache) -> dict[str, dict]:
    """Per PMID: cd (icite_nok), n_i, n_j, n_refs_icite, n_citers_5y."""
    out: dict[str, dict] = {}
    for p in papers:
        pmid = str(p["pmid"])
        rec = icite.get(pmid)
        if rec is None:
            out[pmid] = {"cd": None, "n_i": None, "n_j": None,
                         "n_refs_icite": None, "n_citers_5y": None}
            continue
        focal_refs = {str(x) for x in rec.get("references") or []}
        # iCite lists some papers among their own references; that would force cd to -1.
        focal_refs.discard(pmid)
        year = p.get("year") or rec.get("year")
        citers, _ = window_citers(rec.get("citedByPmidsByYear"), year)
        citer_refs = {}
        for c in citers:
            crec = cache.get(c)
            if crec is None or crec.get("missing"):
                # dated by iCite as a citer, so it cites the focal paper;
                # without its reference list it cannot be classified.
                continue
            citer_refs[c] = set(crec.get("references") or [])
        res = disruption_nok(focal_refs, citer_refs)
        res.update({"n_refs_icite": len(focal_refs),
                    "n_citers_5y": len(citers),
                    "n_citers_unfetched": len(citers) - len(citer_refs)})
        out[pmid] = res
    return out


# ---------------------------------------------------------------- OpenAlex


class Budget(Exception):
    """OpenAlex daily credit budget exhausted."""


class OpenAlex:
    def __init__(self, cache_dir: Path, get=None, min_interval: float = 0.15):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._get = get or self._http
        self.min_interval = min_interval
        self.remaining: int | None = None
        self.requests = 0
        self.session = requests.Session()

    def _http(self, params: dict) -> tuple[dict, dict]:
        resp = http_get(OPENALEX_URL, params, sleep=self.min_interval,
                        session=self.session)
        headers = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code == 429:
            return {"_status": 429}, headers
        return resp.json(), headers

    def call(self, params: dict) -> dict:
        if self.remaining is not None and self.remaining <= OPENALEX_RESERVE:
            raise Budget(f"remaining credits {self.remaining}")
        body, headers = self._get(params)
        self.requests += 1
        rem = headers.get("x-ratelimit-remaining")
        if rem is not None:
            try:
                self.remaining = int(float(rem))
            except ValueError:
                pass
        if body.get("_status") == 429:
            self.remaining = 0
            raise Budget("HTTP 429")
        return body

    # focal lookup -------------------------------------------------------
    def focal(self, pmids: list[str]) -> JsonlCache:
        cache = JsonlCache(self.cache_dir / "focal_by_pmid.jsonl")
        todo = [p for p in dict.fromkeys(pmids) if p not in cache]
        logger.info("OpenAlex focal: %d cached, %d to fetch", len(pmids) - len(todo), len(todo))
        for i in range(0, len(todo), OPENALEX_BATCH):
            b = todo[i:i + OPENALEX_BATCH]
            body = self.call({
                "filter": "ids.pmid:" + "|".join(b),
                "select": "id,ids,publication_year,referenced_works,cited_by_count",
                "per-page": 200,
            })
            by_pmid: dict[str, list[dict]] = {}
            for w in body.get("results", []):
                pm = pmid_from_openalex(w)
                if pm:
                    by_pmid.setdefault(pm, []).append(w)
            recs = []
            for p in b:
                w = pick_openalex_work(by_pmid.get(p, []))
                if w is None:
                    recs.append({"pmid": p, "found": False})
                else:
                    recs.append({"pmid": p, "found": True, "id": short_id(w["id"]),
                                 "publication_year": w.get("publication_year"),
                                 "cited_by_count": w.get("cited_by_count"),
                                 "n_candidates": len(by_pmid[p]),
                                 "referenced_works": [short_id(x) for x in w.get("referenced_works") or []]})
            cache.add_many(recs)
            if (i // OPENALEX_BATCH) % 20 == 0:
                logger.info("OpenAlex focal %d/%d (credits left %s)", i + len(b), len(todo), self.remaining)
        return cache

    # citers of one work -------------------------------------------------
    def citers(self, work_id: str, year: int, cap: int = OPENALEX_CITER_CAP) -> dict:
        path = self.cache_dir / "citers" / f"{work_id}_{year}_{year + WINDOW}.json"
        if path.exists():
            return json.loads(path.read_text())
        out: dict[str, list[str]] = {}
        cursor = "*"
        total = None
        while cursor and len(out) < cap:
            body = self.call({
                "filter": f"cites:{work_id},publication_year:{year}-{year + WINDOW}",
                "select": "id,referenced_works",
                "per-page": 200,
                "cursor": cursor,
            })
            total = (body.get("meta") or {}).get("count", total)
            for w in body.get("results", []):
                out[short_id(w["id"])] = [short_id(x) for x in w.get("referenced_works") or []]
            cursor = (body.get("meta") or {}).get("next_cursor")
            if not body.get("results"):
                break
        rec = {"work": work_id, "window": [year, year + WINDOW], "count": total,
               "capped": bool(total and total > len(out)), "citers": out}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
        return rec


def compute_openalex_validation(papers: list[dict], eligible: set[str],
                                focal: JsonlCache, oa: OpenAlex,
                                max_n: int = VALIDATION_MAX) -> tuple[dict[str, dict], dict]:
    """OpenAlex n_k-free cd for the fixed-order random subsample."""
    years = {str(p["pmid"]): p.get("year") for p in papers}
    order = [p for p in validation_order(eligible)
             if (focal.get(p) or {}).get("found") and (focal.get(p) or {}).get("referenced_works")]
    out: dict[str, dict] = {}
    capped: list[str] = []
    stopped = None
    for pmid in order[:max_n]:
        rec = focal.get(pmid)
        year = years.get(pmid) or rec.get("publication_year")
        try:
            cit = oa.citers(rec["id"], int(year))
        except Budget as exc:
            stopped = str(exc)
            break
        focal_refs = set(rec["referenced_works"])
        citer_refs = {k: set(v) for k, v in cit["citers"].items()}
        res = disruption_nok(focal_refs, citer_refs)
        res["capped"] = cit["capped"]
        if cit["capped"]:
            capped.append(pmid)
        out[pmid] = res
        if len(out) % 25 == 0:
            logger.info("OpenAlex validation %d/%d (credits left %s)", len(out), min(max_n, len(order)), oa.remaining)
    info = {"planned": min(max_n, len(order)), "completed": len(out),
            "stopped_early": stopped, "capped_pmids": capped,
            "order_rule": f"eligible PMIDs sorted numerically, shuffled with random.Random({SEED}); first {max_n} with OpenAlex references"}
    return out, info


# ---------------------------------------------------------------- SciSciNet v1 (auxiliary)


def sciscinet_subset(pmids: set[str], dois: dict[str, str], cache_dir: Path,
                     stream=None) -> dict[str, dict]:
    """Stream SciSciNet v1 link + papers tables and keep rows for our papers.

    Returns {pmid: {"mag_id", "sciscinet_d", "join"}}. Cached.
    """
    path = cache_dir / "papers_subset.json"
    if path.exists():
        return json.loads(path.read_text())
    stream = stream or _stream_lines
    mag_to_pmid: dict[str, str] = {}
    for line in stream(f"{SCISCINET_BASE}/df_magid_pubmedid.tsv"):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() in pmids:
            mag_to_pmid[parts[0].strip()] = parts[1].strip()
    logger.info("SciSciNet: %d PMIDs linked to MAG ids", len(mag_to_pmid))
    doi_to_pmid = {d.lower(): p for p, d in dois.items() if d}
    out: dict[str, dict] = {}
    header = None
    n = 0
    for line in stream(f"{SCISCINET_BASE}/SciSciNet_Papers.tsv"):
        parts = line.rstrip("\n").split("\t")
        if header is None:
            header = {name: i for i, name in enumerate(parts)}
            continue
        n += 1
        if n % 20_000_000 == 0:
            logger.info("SciSciNet papers scanned: %d", n)
        mag = parts[0]
        pmid = mag_to_pmid.get(mag)
        join = "pmid"
        if pmid is None:
            doi = parts[header["DOI"]].lower() if len(parts) > header["DOI"] else ""
            pmid = doi_to_pmid.get(doi) if doi else None
            join = "doi"
        if pmid is None:
            continue
        d = parts[header["Disruption"]] if len(parts) > header["Disruption"] else ""
        rec = {"mag_id": mag, "join": join,
               "sciscinet_d": float(d) if d not in ("", "nan") else None}
        prev = out.get(pmid)
        if prev is None or (prev["join"] == "doi" and join == "pmid"):
            out[pmid] = rec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def _stream_lines(url: str):
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(chunk_size=1 << 20, decode_unicode=True):
            if line is not None:
                yield line


# ---------------------------------------------------------------- main


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]  # noqa: E731
    return {"n": len(v), "mean": sum(v) / len(v), "p10": q(0.1), "median": q(0.5),
            "p90": q(0.9), "min": v[0], "max": v[-1]}


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def build_rows(papers, icite, icite_cd, focal, oa_cd, ssn) -> list[dict]:
    rows = []
    for p in papers:
        pmid = str(p["pmid"])
        rec = icite.get(pmid) or {}
        ic = icite_cd.get(pmid) or {}
        fo = focal.get(pmid) if focal is not None else None
        n_refs_oa = len(fo["referenced_works"]) if fo and fo.get("found") else None
        cd = ic.get("cd")
        oa = (oa_cd or {}).get(pmid) or {}
        rows.append({
            "pmid": pmid,
            "rcr": rec.get("relative_citation_ratio"),
            "cd_source": "icite_nok" if cd is not None else None,
            "cd": cd,
            "n_refs_openalex": n_refs_oa,
            "n_refs_icite": ic.get("n_refs_icite"),
            "n_citers_5y": ic.get("n_citers_5y"),
            "n_i": ic.get("n_i"),
            "n_j": ic.get("n_j"),
            "openalex_id": fo.get("id") if fo and fo.get("found") else None,
            "cd_openalex_nok": oa.get("cd"),
            "sciscinet_d": ((ssn or {}).get(pmid) or {}).get("sciscinet_d"),
        })
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sciscinet", action="store_true",
                    help="also stream SciSciNet v1 (~18 GB) for the auxiliary sciscinet_d column")
    ap.add_argument("--no-openalex", action="store_true")
    ap.add_argument("--validation-n", type=int, default=VALIDATION_MAX)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    papers = read_jsonl(PAPERS_PATH)
    icite = {str(r["pmid"]): r for r in read_jsonl(ICITE_PATH)}
    logger.info("papers %d, iCite records %d", len(papers), len(icite))

    # iCite disruption ----------------------------------------------------
    citers_all: list[str] = []
    for p in papers:
        rec = icite.get(str(p["pmid"]))
        if rec and rec.get("references"):
            c, _ = window_citers(rec.get("citedByPmidsByYear"), p.get("year") or rec.get("year"))
            citers_all.extend(c)
    cache = JsonlCache(CACHE_DIR / "icite" / "citer_refs.jsonl")
    fetch_citer_refs(sorted(set(citers_all)), cache, workers=args.workers)
    icite_cd = compute_icite_cd(papers, icite, cache)

    # OpenAlex ------------------------------------------------------------
    focal = None
    oa_cd: dict[str, dict] = {}
    oa_info: dict = {"skipped": True}
    if not args.no_openalex:
        oa = OpenAlex(CACHE_DIR / "openalex")
        try:
            focal = oa.focal([str(p["pmid"]) for p in papers])
            eligible = {str(p["pmid"]) for p in papers
                        if (icite.get(str(p["pmid"])) or {}).get("is_research_article")}
            oa_cd, oa_info = compute_openalex_validation(papers, eligible, focal, oa,
                                                         max_n=args.validation_n)
        except Budget as exc:
            logger.warning("OpenAlex budget exhausted during focal lookup: %s", exc)
            focal = JsonlCache(CACHE_DIR / "openalex" / "focal_by_pmid.jsonl")
            oa_info = {"stopped_early": str(exc), "completed": 0}
        oa_info.update({"requests_this_run": oa.requests, "credits_remaining": oa.remaining})

    ssn = None
    if args.sciscinet:
        pm = {str(p["pmid"]) for p in papers}
        dois = {str(p["pmid"]): p.get("doi") for p in papers}
        ssn = sciscinet_subset(pm, dois, CACHE_DIR / "sciscinet")
    elif (CACHE_DIR / "sciscinet" / "papers_subset.json").exists():
        ssn = json.loads((CACHE_DIR / "sciscinet" / "papers_subset.json").read_text())

    rows = build_rows(papers, icite, icite_cd, focal, oa_cd, ssn)
    write_jsonl(OUT_PATH, rows)

    # manifest (marginals of this table only) ----------------------------
    research = {pm for pm, r in icite.items() if r.get("is_research_article")}
    cds = [r["cd"] for r in rows if r["cd"] is not None]
    pair = [(r["cd"], r["cd_openalex_nok"]) for r in rows
            if r["cd"] is not None and r["cd_openalex_nok"] is not None]
    pair_ssn = [(r["cd"], r["sciscinet_d"]) for r in rows
                if r["cd"] is not None and r["sciscinet_d"] is not None]
    no_refs = sum(1 for p in papers if (icite_cd.get(str(p["pmid"])) or {}).get("n_refs_icite") == 0)
    no_citers = sum(1 for v in icite_cd.values()
                    if v.get("n_refs_icite") and v.get("n_citers_5y") == 0)
    unfetched = sum(v.get("n_citers_unfetched") or 0 for v in icite_cd.values())
    manifest = {
        "run": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "n_research_articles": len(research),
        "coverage": "full corpus (no subsample): the iCite route made the 3,000-paper fallback unnecessary",
        "rcr_non_null": sum(1 for r in rows if r["rcr"] is not None),
        "cd": {
            "source": "icite_nok",
            "definition": "(n_i - n_j) / (n_i + n_j) over PubMed citers dated Y..Y+5 (Y = papers.jsonl year); n_j = citers citing >=1 focal reference; references and citer reference lists from iCite (NIH Open Citation Collection)",
            "n_computed": len(cds),
            "n_computed_research_articles": sum(1 for r in rows if r["cd"] is not None and r["pmid"] in research),
            "null_no_icite_record": sum(1 for p in papers if str(p["pmid"]) not in icite),
            "null_zero_references": no_refs,
            "null_no_5y_citers": no_citers,
            "n_unique_citers_fetched": len(set(citers_all)),
            "citer_links_without_icite_refs": unfetched,
            "distribution": _summary(cds),
            "distribution_research_articles": _summary([r["cd"] for r in rows if r["cd"] is not None and r["pmid"] in research]),
            "citer_cap": "none (iCite batch lookups are cheap)",
        },
        "sciscinet": {
            "cd5_available": False,
            "why": "SciSciNet v2 (with CD5) is gated on Hugging Face (HTTP 401, licence acceptance + login). SciSciNet v1 is open but its Disruption column is Wu et al. 2019 D over all citations to the MAG 2021 snapshot (not CD5).",
            "aux_column": "sciscinet_d = SciSciNet v1 Disruption joined via df_magid_pubmedid.tsv (PMID) or DOI; never used as cd",
            "n_joined": len(ssn) if ssn is not None else None,
            "n_with_d": sum(1 for r in rows if r["sciscinet_d"] is not None),
            "spearman_cd_vs_sciscinet_d": _spearman(*zip(*pair_ssn)) if len(pair_ssn) >= 3 else None,
        },
        "openalex": {
            "why_not_primary": "anonymous OpenAlex limit is 1000 credits/day (x-ratelimit-limit 1000, 1 credit per list call); 5-year citer paging for ~12k papers needs >12k calls",
            "n_refs_openalex_non_null": sum(1 for r in rows if r["n_refs_openalex"] is not None),
            "validation": oa_info,
            "validation_citer_cap": OPENALEX_CITER_CAP,
            "n_pairs_icite_vs_openalex_cd": len(pair),
            "spearman_icite_vs_openalex_cd": _spearman(*zip(*pair)) if len(pair) >= 3 else None,
            "mean_abs_diff_icite_vs_openalex_cd": (sum(abs(a - b) for a, b in pair) / len(pair)) if pair else None,
        },
        "blinding": "reads papers.jsonl and icite.jsonl only; no outcome_b/features; no associations",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s (%d rows, cd computed for %d)", OUT_PATH, len(rows), len(cds))
    return manifest


if __name__ == "__main__":
    main()
