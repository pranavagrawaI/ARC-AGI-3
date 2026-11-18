"""Six actual full-image HTTP cases, bounded by parent-owned absolute deadlines.

No replay-rate measurement. A child preserves bytes while reading either success
or error bodies; the parent kills a stuck/trickling child at its absolute cutoff.
"""
import ast
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def one(endpoint,body_path,output,rid):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    url=urllib.parse.urlsplit(endpoint)
    if url.scheme!='http' or url.hostname!='127.0.0.1' or url.path:
        raise ValueError('only the local serving endpoint is permitted')
    body=Path(body_path).read_bytes();headers={'Content-Type':'application/json','X-Request-Id':rid}
    save(output/'wire.json',dict(body_sha256=hashlib.sha256(body).hexdigest(),headers=headers,bytes=len(body)))
    # The parent owns the absolute deadline while this child blocks, including
    # cold prefill before headers and slow success/error bodies.
    connection=http.client.HTTPConnection(url.hostname,url.port,timeout=None)
    result=dict(rid=rid,started_monotonic=time.monotonic(),status=None,complete=False)
    save(output/'http.json',result)
    try:
        connection.request('POST','/v1/chat/completions',body=body,headers=headers)
        response=connection.getresponse();result['status']=response.status
        result['response_headers']=list(response.getheaders());save(output/'http.json',result)
        with (output/'response.bin').open('wb',buffering=0) as stream:
            total=0
            while True:
                chunk=response.read1(65536)
                if not chunk:break
                total+=len(chunk)
                if total>4*1024**2:raise ValueError('response exceeds 4MiB evidence cap')
                stream.write(chunk)
        if response.length not in (None,0):raise ValueError('truncated HTTP Content-Length')
        result.update(complete=True,response_bytes=total)
    except Exception as error:
        result['error']=repr(error)
    finally:
        connection.close();result['ended_monotonic']=time.monotonic()
        result['elapsed_s']=result['ended_monotonic']-result['started_monotonic']
        save(output/'http.json',result)
    return 0 if result['complete'] else 1


def assess(output,case):
    output=Path(output)
    result=dict(key=case['key'],images=case['images'],ok=False,code_ok=False)
    try:
        receipt=json.loads((output/'http.json').read_text())
        wire=json.loads((output/'wire.json').read_text())
        attempt=json.loads((output/'attempt.json').read_text())
        if hashlib.sha256((output/'request.json').read_bytes()).hexdigest()!=case['request_sha256']:
            raise ValueError('raw outgoing body mismatch')
        if wire['body_sha256']!=case['request_sha256']:raise ValueError('actual wire hash mismatch')
        result.update(http=receipt)
        if (output/'cutoff.json').exists():result['cutoff']=json.loads((output/'cutoff.json').read_text())
        if not receipt['complete'] or receipt['status']!=200:raise ValueError('HTTP failure or incomplete response')
        if not (receipt['rid']==wire['headers']['X-Request-Id']==attempt['rid']):raise ValueError('request identity join')
        start=receipt['started_monotonic'];end=receipt['ended_monotonic']
        if not all(type(v) in (int,float) and math.isfinite(v) for v in (start,end,attempt['deadline_monotonic'])):
            raise ValueError('nonfinite request clock')
        if not start<=end<=attempt['deadline_monotonic']:raise ValueError('late or reversed response clock')
        raw=json.loads((output/'response.bin').read_bytes());usage=raw['usage']
        prompt=usage['prompt_tokens'];completion=usage['completion_tokens']
        if type(prompt)!=int or prompt<=0 or type(completion)!=int or not 0<completion<=256:
            raise ValueError('invalid raw usage')
        if not isinstance(raw['id'],str) or not raw['id']:raise ValueError('missing response identity')
        if len(raw['choices'])!=1:raise ValueError('expected one choice')
        choice=raw['choices'][0]
        if choice['finish_reason'] not in ('stop','length','tool_calls'):raise ValueError('unexpected finish reason')
        for call in choice['message'].get('tool_calls',[]):
            function=call['function']
            if function.get('name')!='python':continue
            arguments=function['arguments']
            if isinstance(arguments,str):arguments=json.loads(arguments)
            code=arguments.get('code')
            if isinstance(code,str) and code.strip():
                ast.parse(code);result['code_ok']=True
        expected=case.get('local_prompt_tokens')
        result.update(ok=True,usage=usage,response_id=raw['id'],finish_reason=choice['finish_reason'],
                      expected_local_prompt_tokens=expected,
                      prompt_count_matches_local=prompt==expected if expected is not None else None)
    except (OSError,ValueError,TypeError,KeyError,SyntaxError,AttributeError) as error:
        result['issue']=str(error)
    return result


def run(cases,endpoint,output,deadline,per_request=300.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pending=list(cases);active=[];results=[];failed=False;admission_stopped=False
    try:
        while pending or active:
            now=time.monotonic()
            # Headers are an earlier failure signal than a completed child:
            # do not admit replacements while a known error body drains.
            for job in active:
                try:
                    receipt=json.loads((job['directory']/'http.json').read_text())
                    bad=(receipt.get('status') is not None and receipt['status']!=200) or bool(receipt.get('error'))
                except FileNotFoundError:bad=False
                except (ValueError,TypeError,AttributeError):bad=True
                if bad and not admission_stopped:
                    admission_stopped=True
                    save(output/'admission-stop.json',dict(key=job['case']['key'],observed_monotonic=now,
                                                         reason='HTTP failure observed before body completion'))
            while pending and len(active)<2 and now<deadline and not admission_stopped:
                case=pending.pop(0);directory=output/case['key'].replace(':','-')
                directory.mkdir(exist_ok=False)
                # Preserve actual outgoing bytes inside the evidence directory.
                body=Path(case['request_path']).read_bytes()
                if hashlib.sha256(body).hexdigest()!=case['request_sha256']:raise ValueError('case request changed')
                request=directory/'request.json';request.write_bytes(body)
                rid='serving-feas7-'+case['key'].replace(':','-')
                process=subprocess.Popen([sys.executable,'-u',__file__,'one',endpoint,str(request),str(directory),rid],
                                         stdin=subprocess.DEVNULL)
                active.append(dict(process=process,case=case,directory=directory,started=now,cutoff=min(deadline,now+per_request)))
                save(directory/'attempt.json',dict(rid=rid,pid=process.pid,started_monotonic=now,
                                                  deadline_monotonic=min(deadline,now+per_request)))
            for job in list(active):
                process=job['process']
                if process.poll() is None and time.monotonic()>=job['cutoff']:
                    process.kill();process.wait(timeout=5)
                    save(job['directory']/'cutoff.json',dict(absolute_cutoff=True,at=time.monotonic(),
                                                           elapsed_s=time.monotonic()-job['started'],returncode=process.returncode))
                if process.poll() is not None:
                    result=assess(job['directory'],job['case']);results.append(result);active.remove(job)
                    save(output/'progress.json',results)
                    if not result['ok']:
                        failed=True;break
            if failed:break
            if not active and (admission_stopped or time.monotonic()>=deadline):break
            time.sleep(.02)
    finally:
        for job in active:
            if job['process'].poll() is None:
                job['process'].kill()
                save(job['directory']/'cutoff.json',dict(cancelled_after_failure=True,at=time.monotonic(),
                                                        elapsed_s=time.monotonic()-job['started']))
            job['process'].wait(timeout=5)
            results.append(assess(job['directory'],job['case']))
        save(output/'progress.json',results)
    codes=sum(row['code_ok'] for row in results)
    functional=len(results)==len(cases) and all(row['ok'] for row in results) and codes>=min(3,len(cases))
    if functional:functional=len({row['response_id'] for row in results})==len(results)
    fit=functional and all(row['prompt_count_matches_local'] is True for row in results)
    status='FIT_FUNCTIONAL_ONLY' if fit else ('FUNCTIONAL_PROMPT_MISMATCH' if functional else 'FEASIBILITY_FAILED')
    verdict=dict(status=status,results=results,long_input_fit_qualified=fit,
                 admission_stopped_on_headers=admission_stopped,
                 completed=len(results),unattempted=[row['key'] for row in pending],code_cases=codes,
                 rate=None,rate_comparison_qualified=False,adoption=False,
                 processor_semantics='UNQUALIFIED: installed identity/usage only; no native render parity claim')
    save(output/'verdict.json',verdict)
    return verdict


def main():
    if sys.argv[1]=='one':return one(*sys.argv[2:])
    bundle=json.loads(Path(sys.argv[2]).read_text());cases=bundle['cases']
    output=Path(os.environ['SCREEN_OUTPUT'])/'candidate-functional'
    deadline=float(os.environ['SCREEN_DEADLINE_MONOTONIC'])-5.
    if sys.argv[1]=='run':
        return 0 if run(cases,sys.argv[3],output,deadline,per_request=bundle['per_request_seconds'])['status']=='FIT_FUNCTIONAL_ONLY' else 1
    if sys.argv[1]=='qualify':
        rows=[assess(output/case['key'].replace(':','-'),case) for case in cases]
        ids=[row.get('response_id') for row in rows]
        passed=(len(cases)==6 and all(row['ok'] and row['prompt_count_matches_local'] is True for row in rows)
                and sum(row['code_ok'] for row in rows)>=3
                and len(set(ids))==len(ids))
        save(output/'raw-readback-verdict.json',dict(status='FIT_FUNCTIONAL_ONLY' if passed else 'FEASIBILITY_FAILED',
             results=rows,rate=None,rate_comparison_qualified=False,adoption=False,
             native_render_identity_qualified=False))
        return 0 if passed else 1
    raise ValueError('unknown client operation')


if __name__=='__main__':sys.exit(main())
