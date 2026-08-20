"""ORACLE COACH ONLY: download privileged ARC game source for offline analysis.

These files are not admissible contestant observations or contestant priors.
They must not enter solver prompts, runtime workspaces, policy development, or
contestant performance claims except in an explicitly oracle-qualified study.

Read-only against the ARC API. Deliberately does NOT go through Arcade.make():
that opens a scorecard and resets the game, which charges an attempt. Only three
GET endpoints are used, mirroring arc_agi.base._download_game:

    GET /api/games                          -> [{game_id, title, ...}, ...]
    GET /api/games/{game_id}                -> metadata dict
    GET /api/games/{game_id}-{version}/source -> python source text

Layout matches what the harness expects: environment_files/{short_id}/{version}/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

BASE_URL = os.getenv("ARC_BASE_URL", "https://three.arcprize.org")
ENV_DIR = Path(__file__).resolve().parent.parent / "environment_files"


def main() -> int:
    print("ORACLE COACH ONLY: privileged game source; do not use as contestant input.")
    api_key = os.getenv("ARC_API_KEY")
    if not api_key:
        print("ARC_API_KEY is not set; source .env first", file=sys.stderr)
        return 2

    session = requests.Session()
    session.headers.update({"X-Api-Key": api_key, "Accept": "application/json"})

    games = get(session, f"{BASE_URL}/api/games").json()
    print(f"listed {len(games)} games")

    ok = 0
    for entry in games:
        full_id = entry.get("game_id", "")
        if not full_id:
            print(f"  skip: entry without game_id: {entry}")
            continue
        try:
            pull_one(session, full_id)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - one bad game must not abort the sweep
            print(f"  FAIL {full_id}: {exc}")

    print(f"\n{ok}/{len(games)} downloaded into {ENV_DIR}")
    return 0 if ok == len(games) else 1


def pull_one(session: requests.Session, full_id: str) -> None:
    """Fetch metadata + source for one game and write it under environment_files/."""
    short_id, _, version = full_id.partition("-")
    metadata = get(session, f"{BASE_URL}/api/games/{full_id}").json()
    version = version or metadata.get("version") or ""

    env_dir = ENV_DIR / short_id / version if version else ENV_DIR / short_id
    env_dir.mkdir(parents=True, exist_ok=True)

    source = get(session, f"{BASE_URL}/api/games/{full_id}/source").text

    # class_name is how arc_agi names the file; fall back the same way it does.
    class_name = metadata.get("class_name") or (short_id[0].upper() + short_id[1:])
    (env_dir / f"{class_name.lower()}.py").write_text(source, encoding="utf-8")

    metadata.setdefault("tags", [])
    metadata.setdefault("baseline_actions", [])
    metadata["local_dir"] = str(env_dir)
    metadata["date_downloaded"] = datetime.now(UTC).isoformat()
    (env_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    title = metadata.get("title", "?")
    print(f"  {short_id} {version}  {len(source):>7,} bytes  {title}")


def get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    raise SystemExit(main())
