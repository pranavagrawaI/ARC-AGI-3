"""Single-attempt launch with immutable identity checks and budget reservation."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id',choices=['cpu-q5','gpu-q5','cpu-q6','gpu-q6'])
    args=parser.parse_args(); directory=HERE/args.run_id
    identity=json.loads((directory/'identity.json').read_text())
    for name,key in [('kernel/tycho-dfranzen.ipynb','notebook_sha256'),
                     ('kernel/kernel-metadata.json','metadata_sha256'),
                     ('manifest.json','source_manifest_sha256'),
                     ('source-bundle/hybrid-runtime.zlib','source_payload_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==identity[key],name
    assert not (directory/'launch-attempt.json').exists(),'Run already attempted; inspect receipt, do not retry'
    metadata=json.loads((directory/'kernel/kernel-metadata.json').read_text())
    if args.run_id.startswith('gpu-'):
        cpu=HERE/args.run_id.replace('gpu-','cpu-',1)
        complete=json.loads((cpu/'output/qualification-complete.json').read_text())
        assert complete['cpu_fixture_passed']
        receipt=json.loads((cpu/'collection-receipt.json').read_text())
        assert receipt['critical_complete']
        assert json.loads((cpu/'identity.json').read_text())['source_payload_sha256']==identity['source_payload_sha256']
        audit=json.loads((cpu/'AUTOMATIC-AUDIT.json').read_text())
        assert audit['receipt_reconciliation']=='passed',audit['issues']
        if args.run_id=='gpu-q6':
            assert complete['actual_gameapi_games']==['ar25','ls20','sb26']
            assert complete['engine_actions']==3 and complete['native_segmentation_passed']
    budget_path=HERE/'budget.json';budget=json.loads(budget_path.read_text())
    count=identity['gpu_count'];hours=count*identity['hard_timeout_seconds']/3600
    assert budget['unreserved_gpu_hours']>=hours
    attempt={'run_id':args.run_id,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'kernel':metadata['id'],'session_timeout_seconds':identity['hard_timeout_seconds'],
             'gpu_count':count,'state':'attempting; do not relaunch if response is lost'}
    (directory/'launch-attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    if count:
        budget['run_reservations'].append({'run_id':args.run_id,'gpu_count':count,
            'hard_timeout_seconds':identity['hard_timeout_seconds'],'reserved_gpu_hours':hours,
            'kernel':metadata['id'],'state':'attempting'})
        budget['unreserved_gpu_hours']-=hours
        budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    api=KaggleApi();api.authenticate()
    try:
        response=api.kernels_push(str(directory/'kernel'),timeout=str(identity['hard_timeout_seconds']))
        result=json.loads(str(response))
        (directory/'launch-response.json').write_text(json.dumps(result,indent=2)+'\n')
        if result.get('error'): raise RuntimeError('Kaggle rejected saved kernel')
        attempt.update(state='accepted',response=result)
        if count:
            budget['run_reservations'][-1].update(state='accepted',version=result['versionNumber'],
                                               accepted_at=attempt['utc'])
            budget_path.write_text(json.dumps(budget,indent=2)+'\n')
        print(json.dumps(result))
    except Exception as error:
        attempt.update(state='launch error; remote acceptance unresolved until inspected',
                       error_type=type(error).__name__)
        raise
    finally:
        (directory/'launch-attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')


if __name__=='__main__':main()
