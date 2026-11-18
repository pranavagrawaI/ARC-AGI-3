"""Read out screen-2/c16 against screen-2/EXPERIMENT-CARD.md. CPU only.
python3 -B analyze_c16.py   -> prints and writes screen-2/RESULT.json"""
import json
from pathlib import Path
import analyze as a

HERE = Path(__file__).resolve().parent
REF = {'levels': 80, 'agg_tok_s': 823.2}  # screen-1 merge RESULT.json
a.STREAMS = 16


def readout(out):
    m, s = a.arm(out), a.serving(out)
    target = json.loads((out / 'pruning/target.json').read_text())
    draft = json.loads((out / 'pruning/draft.json').read_text())
    crashed = any(x not in {'won', 'gave_up', 'cancelled', 'in_progress'} for x in m['states'])
    mechanism = {
        'target_448': bool(target.get('complete')) and target.get('target_experts') == 448,
        'draft_512': draft.get('draft_experts') == 512,
        'gate_16_streams': s['gate_14'] and s['running_max'] == 16,  # key name from analyze.serving; tests a.STREAMS
        'throughput_ge_1.04x_c14': s['agg_tok_s_0_60'] >= 1.04 * REF['agg_tok_s'],
        'kv_p90_le_0.95': s['kv_usage_p90'] <= 0.95,
        'retractions_le_5': s['retractions'] <= 5,
        'preload_zero': m['preload_name_errors'] == 0,
        'out16k_applied': m['tool_calls_between_caps'] > 0 or (bool(m['cutoff_tokens']) and set(m['cutoff_tokens']) == {a.NEW_CAP}),
        'no_crashed_games': not crashed and m['games'] == 25,
    }
    floor = REF['levels'] - a.NO_HARM
    no_harm = m['levels'] >= floor
    if all(mechanism.values()):
        verdict = 'PASS' if no_harm else 'NO_HARM_FAIL: decide'
    elif not (mechanism['kv_p90_le_0.95'] and mechanism['retractions_le_5']):
        verdict = 'MECHANISM_FAIL: KV overflow at 16 streams; c14 is the ceiling'
    elif not mechanism['throughput_ge_1.04x_c14']:
        verdict = 'MECHANISM_FAIL: throughput does not scale; stream lever closed'
    else:
        verdict = 'MECHANISM_FAIL: inspect'
    return {'verdict': verdict, 'no_harm': {'levels': m['levels'], 'reference': REF, 'floor': floor, 'pass': no_harm},
            'mechanism': mechanism, 'serving': s, 'arm': m}


def main():
    result = readout(HERE / 'screen-2/c16/output')
    (HERE / 'screen-2/RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'no_harm', 'mechanism', 'serving')}, indent=2))


if __name__ == '__main__':
    main()
