"""Submit one hidden draw, at most once. Safe to re-run (timer retries, wake-from-sleep).

  submit_draw.py DRAW KERNEL VERSION MESSAGE [--not-before YYYY-MM-DD]

Steps: refresh the OAuth token -> skip if a submission tagged "Hidden draw DRAW" already exists ->
skip if no slot remains today or the UTC date is before --not-before -> record quota -> submit ->
write receipt-DRAW.json. Every outcome, including skips, is appended to draws.log.
"""
import datetime as dt, json, subprocess, sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials

HERE = Path(__file__).parent
COMP = "arc-prize-2026-arc-agi-3"
KAGGLE = "/home/pranav/.local/bin/kaggle"


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line)
    with open(HERE / "draws.log", "a") as f:
        f.write(line + "\n")


def kaggle(*args):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=180).stdout


def main():
    draw, kernel, version, message = sys.argv[1:5]
    not_before = sys.argv[sys.argv.index("--not-before") + 1] if "--not-before" in sys.argv else None
    tag = f"Hidden draw {draw} "
    today = dt.datetime.now(dt.timezone.utc).date()
    if not_before and today < dt.date.fromisoformat(not_before):
        log(f"{draw}: skip, UTC date {today} is before {not_before}"); return
    KaggleCredentials.load(KaggleClient()).refresh_access_token()
    if tag in kaggle("competitions", "submissions", COMP):
        log(f"{draw}: skip, already submitted"); return
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
    main()
