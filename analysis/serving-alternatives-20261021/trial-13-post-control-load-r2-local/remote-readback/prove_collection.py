"""Standalone read-only collection inventory for a failed/unreached release path."""
import argparse,base64,hashlib,json,time,zipfile
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import transport,artifacts

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def refs(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes','complete'}<=value.keys():yield value
        else:
            for item in value.values():yield from refs(item)
    elif isinstance(value,list):
        for item in value:yield from refs(item)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
    freeze=json.loads((HERE/'freeze.json').read_text())
    for name,h in freeze['files'].items():assert sha(HERE/name)==h,name
    for name,h in freeze['references'].items():assert sha(name)==h,name
    arc=HERE/'remote-readback/archive-v1-kernel140441469';receipt=json.loads((arc/'ARCHIVE.json').read_text());assert receipt['complete'] and receipt['CRC_verified']
    assert sha(arc/'output.zip')==receipt['archive_sha256'];inventory=[]
    with zipfile.ZipFile(arc/'output.zip') as zipped:
        for info in zipped.infolist():
            if info.is_dir():continue
            data=zipped.read(info);name=Path(info.filename);assert not name.is_absolute() and '..' not in name.parts
            assert (arc/'members'/name).read_bytes()==data,name
            inventory.append(dict(name=str(name),bytes=len(data),CRC=info.CRC,sha256=hashlib.sha256(data).hexdigest()))
    parent=arc/'members/serving-post-control-load13';root=parent/'output';spec=json.loads((HERE/'packet-spec.json').read_text());attempt=json.loads((HERE/'launch-attempt.json').read_text())
    terminal=HERE/'remote-readback/poll-20261023T191752574074Z';readback=json.loads((terminal/'READBACK.json').read_text());meta=json.loads((terminal/'metadata.json').read_text())
    assert meta['id']==140441469 and meta['currentVersionNumber']==1 and meta['ref']==spec['metadata']['id']
    source=(terminal/'source.ipynb').read_bytes();armed=(HERE/'kernel'/spec['metadata']['code_file']).read_bytes()
    assert transport.cell_source(json.loads(source))==transport.cell_source(json.loads(armed));inner=transport.decode(source);assert transport.sha(inner)==attempt['inner_notebook_sha']
    values=transport.literals(transport.cell_source(json.loads(inner)));manifest=json.loads((HERE/'review-packet.json').read_text());assert values['CONFIG']==spec['config']
    assert {k:v['sha256'] for k,v in values['PAYLOAD'].items()}==manifest['files']
    for name,record in values['PAYLOAD'].items():assert (parent/name).read_bytes()==base64.b64decode(record['base64'],validate=True),name
    assignment=json.loads((parent/'assignment.json').read_text());assert assignment==values['ASSIGNMENT'];assert assignment['approved_freeze_sha256']==sha(HERE/'freeze.json');assert json.loads((parent/'config.json').read_text())==spec['config']
    result=json.loads((root/'result.json').read_text());events=[json.loads(s) for s in (root/'supervisor.jsonl').read_text().splitlines()];assert [e for e in events if e['event']=='terminal']==[dict(seq=events[-1]['seq'],t=events[-1]['t'],event='terminal',**result)]
    rawrefs=[]
    for value in [json.loads((root/'query-compatibility.json').read_text()),events]:
        for ref in refs(value):artifacts.verify(root/'query-artifacts',ref,time.monotonic()+600);rawrefs.append(ref)
    unique={r['path']:r for r in rawrefs}
    boundaries=[json.loads(s) for s in (root/'host-boundaries.jsonl').read_text().splitlines()];series=[json.loads(s) for s in (root/'host-series.jsonl').read_text().splitlines()];sr=json.loads((root/'host-series.result.json').read_text())
    assert sr['samples']==len(series) and sr['bytes']==(root/'host-series.jsonl').stat().st_size and sr['series_sha256']==sha(root/'host-series.jsonl')
    counts=[]
    for row in boundaries+series:
        parsed={k:int(v) for k,v in (line.split() for line in row['raw']['events'].splitlines())};assert parsed==row['oom_events'];counts.append([parsed['oom'],parsed['oom_kill']])
    quota=readback['typed_quota'];assert readback['status']['status']=='ERROR' and quota['reserved_s']==0 and quota['used_s']==70064.281
    reduced=json.loads((HERE/'remote-readback/derived-original/REDUCTION.json').read_text());assert reduced['status']=='NOT_QUALIFIED' and reduced['ratio'] is None
    proof=dict(status='TERMINAL_COLLECTION_COMPLETE_NOT_QUALIFIED',freeze_sha256=sha(HERE/'freeze.json'),frozen_files=len(freeze['files']),frozen_references=len(freeze['references']),archive_sha256=receipt['archive_sha256'],archive_bytes=receipt['archive_bytes'],CRC_and_extracted_bytes=True,members=len(inventory),exact_payloads=len(values['PAYLOAD']),config_assignment_source_exact=True,kernel_id=140441469,version=1,inner_sha256=transport.sha(inner),result=result,terminal_typed_quota=quota,delta_used_s=round(quota['used_s']-68903.943,6),raw_reference_occurrences=len(rawrefs),unique_raw_references=len(unique),observer_series=sr,boundary_count=len(boundaries),oom_counter_pairs=sorted({tuple(c) for c in counts}),ready_events=[e for e in events if e['event']=='ready'],release_events=[e for e in events if e['event'] in ('release_start','release_result')],candidate_spawns=[e for e in events if e['event']=='spawn' and e.get('phase','').startswith('candidate.')],health_probes=len(list(root.glob('control.health-*.stdout'))),absent_receipts=[name for name in ['pre-release.json','release.json','candidate-admission.json','candidate-package'] if not (root/name).exists()],cleanup_qualified=False,slot_release_claim=False,original_reducer_sha256=sha(HERE/'collection_tools/reduce.py'),original_reducer_result_sha256=sha(HERE/'remote-readback/derived-original/REDUCTION.json'),gaps=['Frozen reducer stops early on absent release.json; standalone proof retains terminal/archive/primary evidence without changing verdict','Preliminary observation may have been attempted; no completed pre-release receipt','Actual failing fd numbers/call stack not recorded','Cleanup identity error remains independent of final empty groups','No release or candidate fit evidence'])
    (out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n');(out/'RAW-REFERENCES.json').write_text(json.dumps(list(unique.values()),indent=2)+'\n');(out/'COLLECTION-PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ('status','archive_sha256','members','exact_payloads','frozen_files','frozen_references','unique_raw_references','health_probes','boundary_count')}))
if __name__=='__main__':main()
