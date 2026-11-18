"""Hash-seal launch adapters without mutating the prepared experiment or Git metadata."""
import json
from launch import HERE,now,sha

FILES=['launch.py','kaggle_backend.py','collect.py','execute.py','seal_launch.py','test_launch.py','AUTHORITY.json','LAUNCH-SETUP.md','prepared-manifest.json']


def seal(root=HERE):
    prepared=json.loads((root/'prepared-manifest.json').read_text())
    for name,digest in prepared['files'].items():
        if sha(root/name)!=digest:raise ValueError('Original prepared artifact changed: '+name)
    files={name:sha(root/name) for name in FILES}
    target=root/'launch-freeze.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['files']!=files:raise ValueError('Launch seal already exists for different bytes; preserve it')
        return old
    result={'utc':now(),'run_id':root.name,'base_commit':'ce99ac6caef4a8fa2eb39f37cf0e7233ee157dda','files':files,
            'scope':'unchanged prepared experiment plus one-attempt launch and read-only collection adapters',
            'git_metadata':'read-only in current session; adapter identity is this exclusive SHA256 manifest'}
    with target.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(seal(),indent=2))
