"""CPU exact wire reuse and analytical collection envelope, no model/server/API."""
import hashlib,json,time
from pathlib import Path
import catalog,corpus_base
HERE=Path(__file__).resolve().parent

def qualify(destination):
 destination=Path(destination);destination.mkdir(exist_ok=False);start=time.monotonic()
 catalog.verify_corpus(corpus_base.DATA);cat=catalog.load();rows=[]
 for key,row in cat['records'].items():
  body=catalog.payload_for(row,corpus_base.DATA);raw=catalog.canonical(body);catalog.checked_body(raw,row)
  rows.append(dict(key=key,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),images=len(row['images'])))
 total=sum(row['bytes'] for row in rows)
 # This bounds names independent of successful outcome/observed rate.
 envelope=dict(request_files=2*1369*4,health_files=2*(901+1351),candidate_resource_files=(91+1)*14,
  native_resource_files=(61+31+1)*15,compatibility_files=8,fixed_payload_package_and_phase_allowance=256)
 required=sum(envelope.values());listed=required+1369 # generated canonical wire-bodies, not scientific output dependencies
 report=dict(scope='LOCAL_CPU_WIRE_AND_COLLECTION_ENVELOPE_NOT_GPU',records=len(rows),exact_bodies=True,
  source_png_references=sum(row['images'] for row in rows),catalog_sha256=hashlib.sha256((HERE/'catalog.json').read_bytes()).hexdigest(),
  maximum_names=envelope,required_names_bound=required,required_name_cap=20000,listed_names_bound=listed,listed_name_cap=30000,
  GET_bound=2*required,GET_cap=40000,corpus_body_bytes_one_arm=total,two_arm_full_wire_upper_bytes=4*total+2738*2048,
  total_collection_byte_cap=8*1024**3,cooperative_seconds=7200,concurrency=1,
  byte_limit_note='Fixed corpus wire bytes are measured; response/native/series bytes share the hard8GiB cap. Per-response16MiB worst-case aggregate can exceed this cap. Explicit Content-Length/stream excess stops collection and makes qualification invalid; no omission or residual pass.',
  assumptions='Health retry delay>=2s; resource rejected-sample poll>=2s; original180 shared bounds. Full raw dependencies are mandatory. Source/name expansion beyond these envelopes rejects before GET.',elapsed_seconds=time.monotonic()-start)
 assert len(rows)==1369 and required<=20000 and listed<=30000 and 2*required<=40000
 (destination/'WIRE-BODIES.json').write_text(json.dumps(rows,indent=2)+'\n');(destination/'COLLECTION-ENVELOPE.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 import sys
 print(json.dumps(qualify(sys.argv[1]),indent=2))
