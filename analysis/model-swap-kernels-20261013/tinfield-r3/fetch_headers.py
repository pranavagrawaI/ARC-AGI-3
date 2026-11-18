"""Ranged-fetch real safetensors headers for the base PLE shards and the core shards."""
import json, struct, sys
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest
from kagglesdk.models.types.model_enums import ModelFramework
OUT = Path(__file__).resolve().parent/'kaggle-mounted'
SRC = {'core': ('natnitarach','tinfield-1-nvfp4-probe',ModelFramework.MODEL_FRAMEWORK_PY_TORCH,'core',1),
       'base': ('dfranzen','intel-qwen3.8-flash-next-w4a16-autoround',ModelFramework.MODEL_FRAMEWORK_TRANSFORMERS,'default',1)}
api = KaggleApi(); api.authenticate()
def header(which, fn):
    o,m,fw,inst,v = SRC[which]
    q = ApiDownloadModelInstanceVersionRequest(); q.owner_slug,q.model_slug,q.framework,q.instance_slug,q.version_number=o,m,fw,inst,v; q.path=fn
    with api.build_kaggle_client() as c:
        resp = c.models.model_api_client.download_model_instance_version(q)
        url = resp.url; resp.close()
    n = struct.unpack('<Q', requests.get(url, headers={'Range':'bytes=0-7'}, timeout=60).content[:8])[0]
    raw = requests.get(url, headers={'Range':f'bytes=8-{7+n}'}, timeout=120).content
    assert len(raw) == n, (fn, len(raw), n)
    return n, json.loads(raw)
for which in ['core','base']:
    files = [f['name'] for f in json.loads((OUT/which/'files.json').read_text()) if f['name'].endswith('.safetensors')]
    if which == 'base': files = [f for f in files if 'ple' in f]
    hd = OUT/which/'headers'; hd.mkdir(exist_ok=True)
    for fn in files:
        n, h = header(which, fn); (hd/(fn+'.json')).write_text(json.dumps({'header_bytes': n, 'tensors': h}))
        print(which, fn, n, len(h), flush=True)
