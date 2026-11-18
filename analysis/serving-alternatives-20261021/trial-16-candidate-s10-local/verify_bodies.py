"""Actual current-packet all-body preparation; no render inference/GPU/network."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import replay,catalog,corpus_base
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();started=time.monotonic()
 for module in (replay,catalog,corpus_base):assert Path(module.__file__).resolve().parent==HERE
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)/'bodies';index=replay.prepare(corpus_base.DATA,root);cat=catalog.load();size=0;max_context=0
  for key,row in index.items():
   raw=(root/row['file']).read_bytes();record=cat['records'][key];catalog.checked_body(raw,record);assert row['sha256']==record['prepared_body_sha256'];size+=len(raw);max_context=max(max_context,record['local_prompt']+record['cap'])
  assert len(index)==1369 and max_context<=139264
  report=dict(status='CURRENT_PACKET_EXACT1369_BODIES_ONLY',records=1369,body_bytes=size,max_context_total=max_context,catalog_sha256=hashlib.sha256((HERE/'catalog.json').read_bytes()).hexdigest(),source_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (replay,catalog,corpus_base)},seconds=time.monotonic()-started,gpu_work=False,network_calls=0)
 with a.output.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
