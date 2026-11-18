"""Generate the v3 harness copies with persistent seed and pre-reservation gate."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OWN=ROOT/'experiments/probes/epistemic-critic-local'


def replace_once(text,old,new,label):
    if text.count(old)!=1:raise RuntimeError(f'{label}: expected one occurrence, found {text.count(old)}')
    return text.replace(old,new)


def main():
    hook=(OWN/'live_hook.py').read_text()
    hook=replace_once(hook,
      '"""External, process-local submit pause for the bounded live critic pilot."""',
      '"""Persistent-prefix submit pause for the system-pivot comparison."""','hook docstring')
    hook=replace_once(hook,"Context(seed, prefix=False)","Context(seed, prefix=True)",'seed placement')
    (OWN/'live_hook_v2.py').write_text(hook)

    harness=(OWN/'live_system_pivot.py').read_text()
    harness=replace_once(harness,
      '"""Matched restarted Actor runs: system-level correction speed and retention."""',
      '"""Matched restarted runs with persistent seed and fail-closed input gate."""','harness docstring')
    harness=harness.replace("own/'live_hook.py'","own/'live_hook_v2.py'")
    if "own/'live_hook.py'" in harness or harness.count("own/'live_hook_v2.py'")!=2:
        raise RuntimeError('hook path replacement failed')
    needle="""        raw=json.dumps(body).encode()
        if len(raw)>1000000:raise AdmissionStop('input byte cap')
        reserve=(len(raw)+4096)*.42/1e6+body['max_tokens']*3/1e6
"""
    gate="""        raw=json.dumps(body).encode()
        if len(raw)>1000000:raise AdmissionStop('input byte cap')
        if purpose=='actor':
            rendered=json.dumps(body.get('messages',[]),ensure_ascii=False)
            markers=('Resume after seven previously executed actions.',
                     'step size 2 rows vertical / 5 cols horizontal',
                     'rows 42-44 cols 39-43, 15 px')
            missing=[marker for marker in markers if marker not in rendered]
            if missing:raise RuntimeError('Actor evidence absent before reservation: '+repr(missing))
        reserve=(len(raw)+4096)*.42/1e6+body['max_tokens']*3/1e6
"""
    harness=replace_once(harness,needle,gate,'pre-reservation marker gate')
    (OWN/'live_system_pivot_v3.py').write_text(harness)


if __name__=='__main__':main()
