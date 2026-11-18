"""Run the v1 matched harness with a fail-closed seed-delivery check."""
import json
from pathlib import Path
import sys
import urllib.request

import live_system_pivot as harness

MARKERS = (
    "Resume after seven previously executed actions.",
    "step size 2 rows vertical / 5 cols horizontal",
    "rows 42-44 cols 39-43, 15 px",
)
_urlopen = urllib.request.urlopen


def checked_urlopen(request, *args, **kwargs):
    """Block paid Actor inference if the prepared evidence is absent."""
    url = getattr(request, "full_url", str(request))
    if url == "https://openrouter.ai/api/v1/chat/completions":
        body = json.loads(request.data)
        # Critic requests deliberately have no Actor tools. Actor requests do.
        if body.get("tools"):
            messages = json.dumps(body.get("messages", []), ensure_ascii=False)
            missing = [marker for marker in MARKERS if marker not in messages]
            if missing:
                raise RuntimeError("Actor evidence absent before paid inference: " + repr(missing))
    return _urlopen(request, *args, **kwargs)


def main():
    if "--fixture" not in sys.argv:
        raise RuntimeError("--fixture is required")
    fixture = Path(sys.argv[sys.argv.index("--fixture") + 1]).resolve()
    seed = (fixture / "seed.txt").read_text()
    missing = [marker for marker in MARKERS if marker not in seed]
    if missing:
        raise RuntimeError("fixture seed is missing required evidence: " + repr(missing))
    if len(seed.encode()) > 10_000:
        raise RuntimeError("fixture seed exceeds the preflight byte ceiling")
    harness.urllib.request.urlopen = checked_urlopen
    harness.main()


if __name__ == "__main__":
    main()
