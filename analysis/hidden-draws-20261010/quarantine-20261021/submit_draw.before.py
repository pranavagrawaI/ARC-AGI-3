"""Submit one hidden draw, at most once. Safe to re-run (timer retries, wake-from-sleep).

  submit_draw.py DRAW KERNEL VERSION MESSAGE [--not-before YYYY-MM-DD] [--requires D11,D12]

Steps: refresh the OAuth token -> skip if a submission tagged "Hidden draw DRAW" already exists ->
with --requires, hold unless every named draw is COMPLETE with a score above 0 (exit 3 if one failed,
4 if one is still pending; daily_draw.py decides the fallback) -> skip if no slot remains today or the
UTC date is before --not-before -> record quota -> submit -> write receipt-DRAW.json.
Every outcome, including skips, is appended to draws.log.
"""
import csv, datetime as dt, io, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
COMP = "arc-prize-2026-arc-agi-3"
KAGGLE = "/home/pranav/.local/bin/kaggle"
REQUIRES_FAILED, REQUIRES_PENDING = 3, 4


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line)
    with open(HERE / "draws.log", "a") as f:
        f.write(line + "\n")


def kaggle(*args):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=180).stdout


def requirement_state(submissions_csv, draws):
    """'met' if every draw is COMPLETE with a public score above 0, 'failed' if one ended otherwise, 'pending' if one
    is still running or absent, 'unknown' if the listing has no draws at all (a failed CLI call, not a pending draw)."""
    lines = submissions_csv.splitlines(True)
    start = next((i for i, l in enumerate(lines) if l.startswith("ref,")), len(lines))  # skip CLI warnings
    rows = {r["description"].split(" ")[2]: r for r in csv.DictReader(io.StringIO("".join(lines[start:])))
            if r["description"].startswith("Hidden draw ")}
    if not rows:
        return "unknown"
    states = []
    for draw in draws:
        r = rows.get(draw)
        status = r["status"].split(".")[-1] if r else "MISSING"
        if status == "COMPLETE":
            states.append("met" if float(r["publicScore"] or 0) > 0 else "failed")
        else:
            states.append("failed" if status in ("ERROR", "CANCELLED") else "pending")
    return "failed" if "failed" in states else "pending" if "pending" in states else "met"


def main():
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    from kaggle.api.kaggle_api_extended import KaggleApi
    draw, kernel, version, message = sys.argv[1:5]
    not_before = sys.argv[sys.argv.index("--not-before") + 1] if "--not-before" in sys.argv else None
    requires = sys.argv[sys.argv.index("--requires") + 1].split(",") if "--requires" in sys.argv else []
    tag = f"Hidden draw {draw} "
    today = dt.datetime.now(dt.timezone.utc).date()
    if not_before and today < dt.date.fromisoformat(not_before):
        log(f"{draw}: skip, UTC date {today} is before {not_before}"); return
    KaggleCredentials.load(KaggleClient()).refresh_access_token()
    if tag in kaggle("competitions", "submissions", COMP):
        log(f"{draw}: skip, already submitted"); return
    if requires:
        state = requirement_state(kaggle("competitions", "submissions", COMP, "--csv"), requires)
        if state != "met":
            log(f"{draw}: hold {kernel} v{version}, requires {','.join(requires)}: {state}")
            return {"failed": REQUIRES_FAILED, "pending": REQUIRES_PENDING}.get(state, 1)  # unknown: retry next hour
    limits = kaggle("competitions", "submission-limits", COMP)
    if "Remaining today: 1" not in limits:
        log(f"{draw}: skip, no slot: {limits.strip()!r}"); return
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as c:
        quota = json.loads(str(c.kernels.kernels_api_client.get_accelerator_quota_statistics()))
    (HERE / f"quota-before-{draw.lower()}.json").write_text(json.dumps(
        {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "quota": quota}, indent=1))
    out = subprocess.run([KAGGLE, "competitions", "submit", COMP, "-k", kernel, "-v", version,
                          "-f", "submission.parquet", "-m", tag + message],
                         capture_output=True, text=True, timeout=300)
    receipt = {"draw": draw, "kernel": kernel, "version": version, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "stdout": out.stdout, "stderr": out.stderr, "returncode": out.returncode,
               "submissions_after": kaggle("competitions", "submissions", COMP).splitlines()[:4]}
    (HERE / f"receipt-{draw.lower()}.json").write_text(json.dumps(receipt, indent=1))
    log(f"{draw}: submitted rc={out.returncode} {out.stdout.strip()[:200]!r}")


if __name__ == "__main__":
    sys.exit(main())
