"""Future authorized collection identity; no API work at import."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
def accepted():
    response=json.loads((HERE/'launch-response.json').read_text());spec=json.loads((HERE/'packet-spec.json').read_text())
    if response.get('error') or any(response.get(k) for k in ('invalidTags','invalidDatasetSources','invalidCompetitionSources','invalidKernelSources','invalidModelSources')):raise ValueError('no clean accepted response')
    if response.get('ref')!='/code/'+spec['metadata']['id'] or response.get('versionNumber')!=1 or type(response.get('kernelId')) is not int:raise ValueError('accepted successor identity/version missing')
    return spec['metadata']['id'],response['kernelId'],response['versionNumber']
