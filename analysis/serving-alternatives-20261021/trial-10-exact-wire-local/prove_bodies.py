"""Real all1369 preparation in a disposable local directory; CPU only."""
import hashlib,json,signal,tempfile,time
from pathlib import Path
import catalog,corpus_base,replay
HERE=Path(__file__).resolve().parent
signal.alarm(120);start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='trial10-prepared-bodies-') as directory:
 root=Path(directory)/'bodies';index=replay.prepare(corpus_base.DATA,root);cat=catalog.load()
 for key,row in index.items():
  raw=(root/row['file']).read_bytes();catalog.checked_body(raw,cat['records'][key]);assert row['sha256']==cat['records'][key]['prepared_body_sha256']
 report=dict(scope='LOCAL_CPU_EXACT_PREPARE_ALL1369_NO_NETWORK_OR_GPU',records=len(index),index_sha256=catalog.digest((root/'index.json').read_bytes()),catalog_sha256=hashlib.sha256((HERE/'catalog.json').read_bytes()).hexdigest(),all_prepared_body_hashes_match_accepted_catalog=True,elapsed_seconds=time.monotonic()-start,temporary_bodies_retained=False)
(HERE/'evidence/all-bodies-prepared.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
