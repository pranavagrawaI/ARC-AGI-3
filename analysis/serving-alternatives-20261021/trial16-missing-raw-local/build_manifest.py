"""Regenerate only the reviewed missing names; never API work."""
import hashlib,json,sys
from binding import HERE,ROOT,ASSESSMENT,CURRENT_PRIMARY,ACTUAL_RESULT

def build():
    gaps=ASSESSMENT/'REQUIRED-GAPS.json'
    assert hashlib.sha256(gaps.read_bytes()).hexdigest()=='e79543e2f7c5d96fb39dfd9a9c237b20367234aeaf17f3596ade1b46b79e27b7'
    value=json.loads(gaps.read_bytes());rows=value['missing'];assert len(rows)==1595
    primary=json.loads((CURRENT_PRIMARY/'PRIMARY.json').read_bytes())
    actual=primary['files']['result.json']
    rows=[dict(row) for row in rows]+[dict(name=ACTUAL_RESULT,conditional=False,expected_sha256=None,expected_bytes=None,
        reasons=['independent actual post-publication result; primary substitute forbidden'],
        comparison_only_primary_sha256=actual['sha256'],comparison_only_primary_bytes=actual['bytes'])]
    origins=json.loads((ASSESSMENT/'ORIGINS.json').read_bytes())
    for row in rows:
        assert row['name']==ACTUAL_RESULT or row['name'] not in origins
        assert not any(item['kind']=='CRC_VERIFIED_SEQUENTIAL_LOCAL_MEMBER' for item in origins.get(row['name'],[]))
        row.pop('origins',None);row.pop('available',None)
    assert len({row['name'] for row in rows})==1596
    return dict(kind='EXACT_TRIAL16_REQUIRED_MISSING_MANIFEST',remote_authorized=False,
        kernel_ref='pranavagrawai/serving-candidate-s10-16-20261024',kernel_id=140493086,version=1,
        original_gaps_sha256=hashlib.sha256(gaps.read_bytes()).hexdigest(),
        expansion='independent actual output/result.json; preserve differing actual bytes, never substitute bundled data',
        bounds=dict(list_pages=100,page_size=100,downloads=1596,bytes=2147483648,wall_seconds=1800,concurrency=1),
        files=sorted(rows,key=lambda row:row['name']))
if __name__=='__main__':
    raw=json.dumps(build(),indent=2).encode()+b'\n'
    if '--check' in sys.argv:assert (HERE/'MISSING-MANIFEST.json').read_bytes()==raw
    else:
        with (HERE/'MISSING-MANIFEST.json').open('xb') as stream:stream.write(raw)
