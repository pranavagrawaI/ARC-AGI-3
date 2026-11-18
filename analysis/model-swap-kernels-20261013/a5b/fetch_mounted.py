"""Fetch the CURRENT Kaggle file listings and small config/index files of the two mounted model instances (target + MTP drafter)."""
import gzip, hashlib, io, json, zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest, ApiListModelInstanceVersionFilesRequest
from kagglesdk.models.types.model_enums import ModelFramework
OUT = Path(__file__).resolve().parent/'kaggle-mounted'
TF = ModelFramework.MODEL_FRAMEWORK_TRANSFORMERS
SRC = {'base': ('dfranzen','intel-qwen3.8-flash-next-w4a16-autoround',TF,'default',1),
       'draft': ('dfranzen','albucino-qwen3-8-flash-next-drafter',TF,'default',1)}
api = KaggleApi(); api.authenticate()
def fetch(src, fn):
    o,m,fw,inst,v = src
    q = ApiDownloadModelInstanceVersionRequest(); q.owner_slug,q.model_slug,q.framework,q.instance_slug,q.version_number=o,m,fw,inst,v; q.path=fn
    with api.build_kaggle_client() as c:
        resp = c.models.model_api_client.download_model_instance_version(q)
        with resp: resp.raise_for_status(); data = resp.content
    if data[:2] == b'\x1f\x8b': data = gzip.decompress(data)
    if data[:2] == b'PK': z = zipfile.ZipFile(io.BytesIO(data)); data = z.read(z.namelist()[0])
    return data
for name, src in SRC.items():
    o,m,fw,inst,v = src
    d = OUT/name; d.mkdir(parents=True, exist_ok=True); files = []; token = None
    while True:
        q = ApiListModelInstanceVersionFilesRequest(); q.owner_slug,q.model_slug,q.framework,q.instance_slug,q.version_number=o,m,fw,inst,v; q.page_size=200
        if token: q.page_token = token
        with api.build_kaggle_client() as c: r = c.models.model_api_client.list_model_instance_version_files(q)
        files += [{'name': f.name, 'size': f.size} for f in r.files]; token = r.next_page_token
        if not token: break
    (d/'files.json').write_text(json.dumps(files, indent=1))
    small = [f['name'] for f in files if f['name'].endswith(('config.json','model.safetensors.index.json'))]
    for fn in small:
        data = fetch(src, fn); p = d/fn; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        print(name, fn, len(data), hashlib.sha256(data).hexdigest()[:16], flush=True)
    print(name, 'files', len(files), 'bytes', sum(f['size'] or 0 for f in files), flush=True)
