"""Dump every field of our competition submissions (read-only)."""
import json, sys
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
c = api.build_kaggle_client().__enter__()
out = []
page = ""
while True:
    r = ApiListSubmissionsRequest(); r.competition_name = "arc-prize-2026-arc-agi-3"; r.page_token = page
    resp = c.competitions.competition_api_client.list_submissions(r)
    for s in resp.submissions:
        out.append({k: str(getattr(s, k)) for k in dir(s) if not k.startswith('_') and not callable(getattr(s, k))})
    page = getattr(resp, "next_page_token", "") or ""
    if not page: break
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print(len(out)); print(sorted(out[0].keys()))
