"""Read out screen-1 against its pre-registered rules (screen-1/EXPERIMENT-CARD.md). CPU only.
python3 -B analyze.py [screen-1]   -> prints and writes screen-1/RESULT.json"""
import importlib.util, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('m2_f1_analyze', HERE.parents[1] / 'm2_f1/experiment/analyze.py')
_f1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_f1)
arm, NEW_CAP = _f1.arm, _f1.NEW_CAP  # F1's counters: levels, preload NameErrors, cutoffs, tempo

STREAMS = 14
POOL = {'f1x s2': 76, 'b1ctl s2': 72, 'b1ctl s1': 65}  # B1-lineage 60-min screens at 10 streams
NO_HARM = 7
F1X_POOL_TOKENS, F1X_AGG_TOK_S = 1_012_736, 697.0  # screen-2 f1x serve.log, minutes 0-60
DECODE = re.compile(r"^\[(\S+ \S+)\] Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+), "
                    r"mamba num: (\d+), mamba usage: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)", re.M)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def serving(out):
    import datetime as dt
    log = (out / 'serve.log').read_text(errors='replace')
    rows = DECODE.findall(log)
    t0 = dt.datetime.strptime(rows[0][0], '%Y-%m-%d %H:%M:%S').timestamp()
    first60 = [r for r in rows if dt.datetime.strptime(r[0], '%Y-%m-%d %H:%M:%S').timestamp() - t0 < 3600 and int(r[1])]
    pool = re.search(r'max_total_num_tokens=(\d+)', log)
    kernel = '\n'.join(p.read_text(errors='replace') for p in out.glob('*.log') if p.name != 'serve.log')
    return {
        'pool_tokens': int(pool.group(1)) if pool else None,
        'running_max': max(int(r[1]) for r in rows),
        'agg_tok_s_0_60': round(sum(float(r[6]) for r in first60) / len(first60), 1),
        'per_stream_tok_s_0_60': round(sum(float(r[6]) / int(r[1]) for r in first60) / len(first60), 1),
        'kv_usage_p50': q([float(r[3]) for r in rows], .5), 'kv_usage_p90': q([float(r[3]) for r in rows], .9),
        'mamba_usage_p90': q([float(r[5]) for r in rows], .9),
        'retractions': sum(1 for l in log.splitlines() if 'retract' in l.lower() and 'server_args=' not in l),
        'gate_14': f'priority gate active: {STREAMS} concurrent streams' in kernel,
        'ready_s': next((int(m) for m in re.findall(r'SERVER READY at (\d+)s', kernel)), None),
    }


def readout(out):
    m, s = arm(out), serving(out)
    target = json.loads((out / 'pruning/target.json').read_text())
    draft = json.loads((out / 'pruning/draft.json').read_text())
    crashed = any(x not in {'won', 'gave_up', 'cancelled', 'in_progress'} for x in m['states'])
    mechanism = {
        'target_448': bool(target.get('complete')) and target.get('target_experts') == 448,
        'draft_512': draft.get('draft_experts') == 512,
        'gate_14_streams': s['gate_14'] and s['running_max'] == STREAMS,
        'pool_above_f1x': (s['pool_tokens'] or 0) > F1X_POOL_TOKENS,
        'throughput_up_10pct': s['agg_tok_s_0_60'] >= 1.10 * F1X_AGG_TOK_S,
        'kv_p90_le_0.90': s['kv_usage_p90'] <= 0.90,
        'retractions_le_5': s['retractions'] <= 5,
        'preload_zero': m['preload_name_errors'] == 0,
        'out16k_applied': m['tool_calls_between_caps'] > 0 or (bool(m['cutoff_tokens']) and set(m['cutoff_tokens']) == {NEW_CAP}),
        'no_crashed_games': not crashed and m['games'] == 25,
    }
    floor = sum(POOL.values()) / len(POOL) - NO_HARM
    no_harm = m['levels'] >= floor
    verdict = ('PASS' if all(mechanism.values()) and no_harm else
               'NO_HARM_FAIL: decide; one arm cannot separate REAP from 14 streams' if all(mechanism.values()) else
               'MECHANISM_FAIL: inspect before any rerun')
    result = {'verdict': verdict, 'no_harm': {'merge_levels': m['levels'], 'pool': POOL, 'floor': floor, 'pass': no_harm},
              'mechanism': mechanism, 'serving': s, 'arm': m,
              'receipts': {'target': {k: target.get(k) for k in ('complete', 'target_experts')}, 'draft': draft.get('draft_experts')}}
    return result


def main():
    screen = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'screen-1')
    result = readout(screen / 'merge' / 'output')
    (screen / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'no_harm', 'mechanism', 'serving')}, indent=2))


if __name__ == '__main__':
    main()
