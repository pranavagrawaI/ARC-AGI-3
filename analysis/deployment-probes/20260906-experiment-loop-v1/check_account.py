import json
import os
import urllib.request
from pathlib import Path

for p in (Path(".env"), Path("/home/pranav/projects/arc-agi-3/.env")):
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ARC_LLM_API_KEY")
assert key, "No configured key"
base = "https://openrouter.ai/api/v1"
result = {}
for endpoint in ("credits", "key", "models"):
    req = urllib.request.Request(
        base + "/" + endpoint, headers={"Authorization": "Bearer " + key}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if endpoint == "models":
        result["models"] = [m for m in data["data"] if m["id"] == "qwen/qwen3.8-27b"]
    elif endpoint == "key":
        result["key_budget"] = {
            k: v
            for k, v in data["data"].items()
            if k in ("limit", "limit_remaining", "usage", "is_free_tier")
        }
    else:
        result[endpoint] = data
Path(
    "analysis/deployment-probes/20260906-experiment-loop-v1/PREFLIGHT.json"
).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
