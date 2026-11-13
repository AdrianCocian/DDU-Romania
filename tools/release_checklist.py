#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
release_checklist.py - is the repository ready for a release that the article can cite?

Checks, without any data other than what is in the repository:
  archive    per-cell file of the published run present and complete; its season lengths are those of the archived
             tables; VERIFICATION.txt reports a complete verification; the numbers printed in the article and the
             archived tables are reproduced from the archived per-cell file
  metadata   .zenodo.json and CITATION.cff are well formed and agree
  hygiene    no credential, local path or e-mail address in a tracked file; no file above 50 MB; nothing uncommitted

Usage
  python tools/release_checklist.py                # everything
  python tools/release_checklist.py --quick        # skip the recomputation of the tables (about one minute less)

Exit code 0 and a last line "READY" when every check passes; otherwise 1 and "NOT READY" with the list of problems.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN = REPO / "results" / "run_20260908_131817_6a83be93"
METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
EXPECTED_ROWS = 2912 * 20 * 4
ZENODO_KEYS = {"title", "upload_type", "description", "creators", "contributors", "keywords", "license", "language",
               "notes", "related_identifiers", "communities", "grants", "version", "access_right", "publication_date",
               "references", "subjects"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".cff", ".csv", ".ini", ".cfg", ".toml", ""}
SECRET = re.compile(r"(?im)^\s*key\s*:\s*[0-9a-f]{8}-?[0-9a-f-]{12,}|cdsapi[_-]?key\s*=\s*\S{12,}")
LOCAL_PATH = re.compile(r"(?i)\b[a-z]:\\{1,2}(users\\{1,2}(?!<)|proiect)|/home/[a-z][a-z0-9_-]+/|/Users/[A-Za-z]")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
PATTERN_EXAMPLES = {"tools/release_checklist.py", "tests/test_release_checklist.py"}   # hold examples of the patterns


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def git(*args: str) -> Tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout


def tracked_files() -> List[Path]:
    rc, out = git("ls-files", "-z")
    return [REPO / f for f in out.split("\0") if f] if rc == 0 else []


# ------------------------------------------------------------------ checks: each returns (ok, detail)
def check_per_cell_file() -> Tuple[bool, str]:
    f = RUN / "season_ALL_YEARS_methods.csv.gz"
    if not f.exists():
        return False, f"{f.relative_to(REPO).as_posix()} is missing: run tools/verify_release.py on the data of the published run"
    df = pd.read_csv(f, usecols=["year", "i", "j", "method", "season_start_doy", "season_end_doy", "season_length",
                                 "total_eip_days"], low_memory=False)
    problems = []
    if len(df) != EXPECTED_ROWS:
        problems.append(f"{len(df)} rows instead of {EXPECTED_ROWS}")
    if sorted(df["method"].unique()) != sorted(METHODS):
        problems.append(f"methods {sorted(df['method'].unique())}")
    if (int(df["year"].min()), int(df["year"].max()), df["year"].nunique()) != (2005, 2024, 20):
        problems.append("years are not 2005-2024")
    ref = pd.read_csv(RUN / "per_cell_classes_ALL_YEARS.csv", usecols=["year", "i", "j"] + METHODS)
    wide = (df.pivot_table(index=["year", "i", "j"], columns="method", values="season_length", aggfunc="first")
            .reindex(columns=METHODS).fillna(0.0).reset_index())
    m = ref.merge(wide, on=["year", "i", "j"], how="outer", suffixes=("_ref", ""), indicator=True)
    one_side = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    differ = sum(int((both[k].to_numpy(float) != both[k + "_ref"].to_numpy(float)).sum()) for k in METHODS)
    if one_side or differ:
        problems.append(f"season lengths differ from per_cell_classes_ALL_YEARS.csv ({differ} values, {one_side} rows on one side)")
    return (not problems), ("; ".join(problems) if problems else
                            f"{len(df):,} rows = 2,912 cells x 20 years x 4 methods; season lengths equal to the archived tables")


def check_verification_record() -> Tuple[bool, str]:
    f = RUN / "VERIFICATION.txt"
    if not f.exists():
        return False, "VERIFICATION.txt is missing: run tools/verify_release.py on the data of the published run"
    text = f.read_text(encoding="utf-8", errors="replace")
    result = next((l.strip() for l in text.splitlines() if l.startswith("result:")), "no result line")
    failed = [l for l in text.splitlines() if l.startswith("[FAIL]")]
    ok = result.startswith("result: complete") and not failed
    return ok, result + (f"; {len(failed)} failed stage(s)" if failed else "")


def check_article_numbers() -> Tuple[bool, str]:
    f = RUN / "season_ALL_YEARS_methods.csv.gz"
    if not f.exists():
        return False, "not run: the per-cell file is missing"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        pack = Path(td) / "pack"
        env = dict(__import__("os").environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", DDU_ROOT=td)
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "step09_manuscript_statistics.py"), "--season-csv", str(f),
                            "--out", str(pack), "--no-excel"], cwd=str(REPO), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode != 0:
            return False, "step09 failed on the archived per-cell file: " + (p.stderr.strip().splitlines() or ["?"])[-1]
        p = subprocess.run([sys.executable, str(HERE / "check_against_manuscript.py"), "--pack", str(pack)], cwd=str(REPO),
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith(("Part A:", "Part B:", "RESULT"))]
        return p.returncode == 0, "; ".join(lines) or "no output"


def check_zenodo_json() -> Tuple[bool, str]:
    f = REPO / ".zenodo.json"
    try:
        z = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"not valid JSON: {e}"
    problems = []
    for key in ("title", "upload_type", "description", "creators", "license"):
        if not z.get(key):
            problems.append(f"'{key}' missing")
    unknown = sorted(set(z) - ZENODO_KEYS)
    if unknown:
        problems.append(f"keys Zenodo does not know: {unknown}")
    if z.get("upload_type") != "software":
        problems.append("upload_type is not 'software'")
    if z.get("license") != "mit":
        problems.append("license must be the lower-case identifier 'mit'")
    for c in z.get("creators", []):
        if "," not in str(c.get("name", "")):
            problems.append(f"creator name not in the form 'Family, Given': {c.get('name')!r}")
        if c.get("orcid") and not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", str(c["orcid"])):
            problems.append(f"ORCID not in the form 0000-0000-0000-0000: {c['orcid']!r}")
    names = "; ".join(str(c.get("name")) for c in z.get("creators", []))
    return (not problems), ("; ".join(problems) if problems else f"creators: {names}; licence {z['license']}")


def check_citation_cff() -> Tuple[bool, str]:
    f = REPO / "CITATION.cff"
    text = f.read_text(encoding="utf-8")
    problems = [f"'{k}' missing" for k in ("cff-version:", "message:", "title:", "authors:", "license:", "version:")
                if not re.search(rf"(?m)^{re.escape(k)}", text)]
    try:
        import yaml  # noqa: WPS433 - optional
        c = yaml.safe_load(text)
        cff_authors = [f"{a.get('family-names')}, {a.get('given-names')}" for a in c.get("authors", [])]
        z = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
        zen_authors = [str(x.get("name")) for x in z.get("creators", [])]
        if cff_authors != zen_authors:
            problems.append(f"authors differ from .zenodo.json: {cff_authors} / {zen_authors}")
        if str(c.get("license")) != "MIT":
            problems.append("license is not MIT")
        detail = f"version {c.get('version')}; authors: {'; '.join(cff_authors)}"
    except ImportError:
        detail = "required keys present (PyYAML not installed: authors not compared with .zenodo.json)"
    except Exception as e:
        problems.append(f"not valid YAML: {e}")
        detail = ""
    return (not problems), ("; ".join(problems) if problems else detail)


def check_tracked_content() -> Tuple[bool, str]:
    files = tracked_files()
    if not files:
        return False, "git ls-files returned nothing (not a git repository?)"
    problems = []
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        if f.name == ".cdsapirc" or f.suffix in {".pem", ".key", ".p12", ".pfx"}:
            problems.append(f"credential file tracked: {rel}")
            continue
        if not f.exists() or f.suffix.lower() not in TEXT_SUFFIXES or f.stat().st_size > 5_000_000:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if SECRET.search(text):
            problems.append(f"looks like an API key: {rel}")
        if rel not in PATTERN_EXAMPLES:
            m = LOCAL_PATH.search(text)
            if m:
                problems.append(f"local path '{m.group(0)}...' in {rel}")
            m = EMAIL.search(text)
            if m:
                problems.append(f"e-mail address in {rel}")
    return (not problems), ("; ".join(problems[:8]) if problems else f"{len(files)} tracked files: no credential, local path or e-mail address")


def check_file_sizes() -> Tuple[bool, str]:
    big = [(f.relative_to(REPO).as_posix(), f.stat().st_size / 1e6) for f in tracked_files() if f.exists() and f.stat().st_size > 50e6]
    total = sum(f.stat().st_size for f in tracked_files() if f.exists()) / 1e6
    return (not big), ("; ".join(f"{n} {s:.0f} MB" for n, s in big) + " (GitHub refuses files above 100 MB)" if big
                       else f"largest tracked file below 50 MB; repository content {total:.1f} MB")


def check_committed() -> Tuple[bool, str]:
    rc, out = git("status", "--porcelain")
    if rc != 0:
        return False, "git status failed"
    lines = [l for l in out.splitlines() if l.strip()]
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    _, head = git("rev-parse", "--short", "HEAD")
    return (not lines), (f"{len(lines)} uncommitted change(s), first: {lines[0].strip()}" if lines
                         else f"branch {branch.strip()}, commit {head.strip()}, nothing uncommitted")


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="Release checklist of the DDU-Romania repository")
    ap.add_argument("--quick", action="store_true", help="skip the recomputation of the tables from the per-cell file")
    a = ap.parse_args(argv)

    checks = [
        ("archive: per-cell file of the published run", check_per_cell_file),
        ("archive: verification record", check_verification_record),
        ("archive: numbers of the article from the archived per-cell file", None if a.quick else check_article_numbers),
        ("metadata: .zenodo.json", check_zenodo_json),
        ("metadata: CITATION.cff", check_citation_cff),
        ("hygiene: content of tracked files", check_tracked_content),
        ("hygiene: file sizes", check_file_sizes),
        ("hygiene: everything committed", check_committed),
    ]
    problems = []
    for title, fn in checks:
        if fn is None:
            print(f"[SKIP] {title} (--quick)")
            continue
        try:
            ok, detail = fn()
        except Exception as e:  # a check that cannot run is a failed check
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"[{' OK ' if ok else 'FAIL'}] {title} - {detail}", flush=True)
        if not ok:
            problems.append(title)
    if problems:
        print(f"\nNOT READY: {len(problems)} check(s) failed")
        for t in problems:
            print("  - " + t)
        return 1
    print("\nREADY" + (" (quick: tables not recomputed)" if a.quick else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
