"""Exclusive seal of all executable and precommitted artifacts."""
import json
from launch import HERE,now,sha


def seal():
    prepared=json.loads((HERE/'prepared-manifest.json').read_text())
    for name,digest in prepared['files'].items():
        if sha(HERE/name)!=digest:raise ValueError('Prepared artifact changed: '+name)
    files={name:sha(HERE/name) for name in ['prepared-manifest.json','SDK-CONTRACT.json']}
    path=HERE/'launch-freeze.json'
    result={'utc':now(),'run_id':HERE.name,'files':files,'scope':'one fixed-capture GPU diagnostic and read-only collection'}
    with path.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(seal(),indent=2))
