"""Fetch current Kaggle file listings plus small metadata files for both mounted model instances."""
import json, sys, hashlib
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest, ApiListModelInstanceVersionFilesRequest
from kagglesdk.models.types.model_enums import ModelFramework
OUT = Path(__file__).resolve().parent/'kaggle-mounted'
SRC = {'core': ('natnitarach','tinfield-1-nvfp4-probe','pytorch','core',1),
       'base': ('dfranzen','intel-qwen3.8-flash-next-w4a16-autoround','transformers','default',1)}
FILES = ['config.json','model.safetensors.index.json','tokenizer.json']
api = KaggleApi(); api.authenticate()
FW = {'pytorch': ModelFramework.MODEL_FRAMEWORK_PY_TORCH, 'transformers': ModelFramework.MODEL_FRAMEWORK_TRANSFORMERS}
for name,(o,m,fw,inst,v) in SRC.items():
    d = OUT/name; d.mkdir(parents=True, exist_ok=True); files = []; token = None
    while True:
        q = ApiListModelInstanceVersionFilesRequest(); q.owner_slug,q.model_slug,q.framework,q.instance_slug,q.version_number=o,m,FW[fw],inst,v; q.page_size=200
        if token: q.page_token = token
        with api.build_kaggle_client() as c: r = c.models.model_api_client.list_model_instance_version_files(q)
        files += [{'name': f.name, 'size': f.size} for f in r.files]; token = r.next_page_token
        if not token: break
    (d/'files.json').write_text(json.dumps(files, indent=1))
    for fn in FILES:
        q = ApiDownloadModelInstanceVersionRequest(); q.owner_slug,q.model_slug,q.framework,q.instance_slug,q.version_number=o,m,FW[fw],inst,v; q.path=fn
        with api.build_kaggle_client() as c:
            resp = c.models.model_api_client.download_model_instance_version(q)
            with resp: resp.raise_for_status(); data = resp.content
        if data[:2] == b'\x1f\x8b':
            import gzip; data = gzip.decompress(data)
        if data[:2] == b'PK':
            import zipfile, io; z = zipfile.ZipFile(io.BytesIO(data)); data = z.read(z.namelist()[0])
        (d/fn).write_bytes(data); print(name, fn, len(data), hashlib.sha256(data).hexdigest()[:16], flush=True)
    print(name, 'files', len(files))
