"""Existing approved loader pattern; never serialize credentials or environment."""
import ast, os
from pathlib import Path

def load_key(*, credential_file_authorized=False):
    key=os.environ.get('OPENROUTER_API_KEY')
    if key:return key,'environment'
    if not credential_file_authorized:return None,'credential-use-approval-pending'
    # Same sole-key parser and path as the prior approved launch_once.py.
    path=Path('/home/pranav/projects/arc-agi-3/.env')
    try:
        with path.open() as stream:
            for line in stream:
                name,separator,value=line.strip().removeprefix('export ').partition('=')
                if separator and name.strip()=='OPENROUTER_API_KEY':
                    value=value.strip()
                    key=ast.literal_eval(value) if value.startswith(('"',"'")) else value
                    return (key,'existing-approved-loader') if isinstance(key,str) and key else (None,'missing')
    except (OSError,ValueError,SyntaxError):return None,'unavailable'
    return None,'missing'

if __name__=='__main__':
    import json
    print(json.dumps({'environment_key_present':bool(os.environ.get('OPENROUTER_API_KEY')),'credential_loader_inert':True}))
