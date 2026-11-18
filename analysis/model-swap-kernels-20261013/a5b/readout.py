"""A5B smoke readout: identical definitions for the smoke and the saved c14 controls.

Clocks (both reported, one is the gate):
  benchmark clock (GATE): wall minutes since benchmark.json start_time. Level times come from
    levels_time.py (unchanged, per-game first-transcript-turn clock) plus each game's first-turn offset
    from benchmark start. In all arms the server is READY before the benchmark starts, and the smoke's
    40-minute soft end is on this clock, so it is uncensored to minute 40 in both arms.
  transcript clock (info): levels_time.py as is. Games 15-25 only get a stream ~10-20 min after start
    (14-stream gate), so in a 40-minute smoke their transcript minute 30 is censored; the control's is not.
Throughput: time-weighted SGLang 'Decode batch' gen tok/s (analyze.py estimator) over benchmark-clock
  minutes 5-30; per-stream = time-weighted gen tok/s / #running-req on the same lines.

  python3 -B readout.py [--controls-only | --dry-run OUTPUT_DIR]
"""
import datetime as dt
import importlib.util
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


levels = module('levels_time', REPO / 'analysis/pruning-transfer-20261011/readout/levels_time.py')
analyze = module('serving_analyze', REPO / 'analysis/model-serving-20261011/analyze.py')
CONTROL = {'c14_wave1': REPO / 'analysis/model-serving-20261011/wave-1/e2-c14/output',
           'c14_pruning_control': REPO / 'analysis/pruning-transfer-20261011/control/output'}
PRIMARY = 'c14_wave1'
HDR = re.compile(r"^--- analysis_step=\d+ \| action=(\d+) \| (\d\d:\d\d:\d\d)", re.M)
MINUTES = [5, 10, 15, 20, 25, 30, 35, 40]
VIEW = '/kaggle/a5b-model-view'


def first_turn_offsets(out, start):
    offsets = {}
    for t in Path(out).glob('transcripts/*.txt'):
        h = HDR.findall(t.read_text(errors='replace'))
        if not h:
            continue
        first = dt.datetime.combine(start.date(), dt.datetime.strptime(h[0][1], '%H:%M:%S').time())
        offsets[t.name[:4]] = ((first - start.replace(microsecond=0)).total_seconds() % 86400) / 60
    return offsets


def serve_window(out, start, lo=5, hi=30):
    rows = []
    for line in (Path(out) / 'serve.log').read_text(errors='replace').splitlines():
        m = analyze.DECODE.match(line)
        if m:
            t = dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            rows.append((t, int(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(7))))
    w = tok = per = 0.0
    acc, run, use = [], [], []
    for (t, r, u, a, g, q), nxt in zip(rows, rows[1:]):
        if not (lo * 60 <= (t - start).total_seconds() < hi * 60):
            continue
        d = min(10.0, max(0.0, (nxt[0] - t).total_seconds()))
        w += d
        tok += g * d
        if r:
            per += g / r * d
        acc.append(a); run.append(r); use.append(u)
    return {'clock': 'benchmark', 'window_min': [lo, hi], 'window_s': round(w), 'decode_lines': len(acc),
            'agg_tok_s': round(tok / w, 1) if w else None, 'per_stream_tok_s': round(per / w, 1) if w else None,
            'accept_len_median': st.median(acc) if acc else None, 'running_median': st.median(run) if run else None,
            'kv_usage_median': st.median(use) if use else None}


def arm(out):
    out = Path(out)
    b = json.loads((out / 'benchmark.json').read_text())
    start = dt.datetime.fromisoformat(b['start_time'])
    curve = levels.run(out)
    off = first_turn_offsets(out, start)
    bench = {g: sorted(m + off[g] for m in ms) for g, ms in curve.items()}
    count = lambda c, t: sum(sum(v <= t for v in ls) for ls in c.values())
    states = {g['game_id']: g['state'] for g in b['game_runs']}
    return {'games': len(b['game_runs']), 'game_ids': sorted(states), 'states': states,
            'state_counts': {s: list(states.values()).count(s) for s in sorted(set(states.values()))},
            'benchmark_start': b['start_time'], 'benchmark_end': b.get('end_time'),
            'levels_benchmark_clock': {str(t): count(bench, t) for t in MINUTES},
            'levels_transcript_clock': {str(t): count(curve, t) for t in MINUTES},
            'first_turn_offset_min': {g: round(v, 2) for g, v in sorted(off.items())},
            'per_game_level_minutes_transcript': curve,
            'serve_5_30_benchmark_clock': serve_window(out, start),
            'serve_5_30_analyze_py': analyze.serve_stats(out, 5, 30),
            'levels_completed_total': sum(g['levels_completed'] or 0 for g in b['game_runs'])}


def logs_text(out):
    text = ''
    for p in sorted(Path(out).glob('*.log')):
        if p.name == 'serve.log':
            continue
        raw = p.read_text(errors='replace')
        try:
            text += ''.join(x.get('data', '') for x in json.loads(raw))
        except Exception:
            text += raw
    return text


def evaluate(out, controls):
    out = Path(out)
    t = arm(out)
    c = controls[PRIMARY]
    text = logs_text(out)
    serve = (out / 'serve.log').read_text(errors='replace')
    finite = json.loads((out / 'model-swap-finite-answer.json').read_text())
    lp = finite['choices'][0]['logprobs']['token_logprobs']
    info = json.loads((out / 'model-swap-server-info.json').read_text())
    asm_path = out / 'model-swap-assembly.json'
    asm = json.loads(asm_path.read_text()) if asm_path.exists() else {}
    rpath = out / 'model-swap-routing.json'
    parsed = None
    if rpath.exists():
        m = re.search(r'A5B_ROUTING_PARSED (\{.*\})', json.loads(rpath.read_text()).get('stdout', ''))
        parsed = json.loads(m.group(1)) if m else None
    s, cs = t['serve_5_30_benchmark_clock'], c['serve_5_30_benchmark_clock']
    tok_ratio = s['agg_tok_s'] / cs['agg_tok_s'] if s['agg_tok_s'] else 0.0
    lv30, clv30 = t['levels_benchmark_clock']['30'], c['levels_benchmark_clock']['30']
    gates = {
        'READY': bool(re.search(r'READY after \d+s', text)) or 'SMOKE_READY at' in text,
        'finite_answers': bool(lp) and any(x is not None for x in lp) and all(x is None or math.isfinite(float(x)) for x in lp),
        'acceptance_ge_2': (s.get('accept_len_median') or 0) >= 2.0,
        'routing_top5_live': (info.get('model_path') == VIEW and asm.get('top_k') == 5 and parsed is not None
                              and parsed['target']['num_experts_per_tok'] == 5 and parsed['target']['path'] == VIEW
                              and parsed['target']['num_experts'] == 512),
        'drafter_unchanged_top10': parsed is not None and parsed['draft']['num_experts_per_tok'] == 10,
        'no_crashed_games': t['games'] == 25 and 'crashed' not in t['states'].values()
                            and set(t['states'].values()) <= {'won', 'gave_up', 'cancelled'},
        'same_25_public_games': t['game_ids'] == c['game_ids'],
        'runtime_c14_config': info.get('max_running_requests') == 14 and info.get('context_length') == 102400
                              and info.get('max_mamba_cache_size') == 84 and 'priority gate active: 14 concurrent streams' in text,
        'runtime_autoround_unchanged': info.get('quantization') == 'auto-round'
                                       and info.get('speculative_draft_model_quantization') == 'compressed-tensors',
        'serving_window_sufficient': s['window_s'] >= 1200,
        'throughput_ge_1p10x': tok_ratio >= 1.10,
        'levels30_ge_0p9x_control': lv30 >= 0.9 * clv30,
    }
    return t, gates, {'tok_ratio_vs_c14_wave1': round(tok_ratio, 3),
                      'tok_ratio_vs_pruning_control': round(s['agg_tok_s'] / controls['c14_pruning_control']['serve_5_30_benchmark_clock']['agg_tok_s'], 3) if s['agg_tok_s'] else None,
                      'per_stream_ratio_vs_c14_wave1': round(s['per_stream_tok_s'] / cs['per_stream_tok_s'], 3) if s['per_stream_tok_s'] else None,
                      'levels30_benchmark_clock': [lv30, clv30, round(lv30 / clv30, 3)],
                      'levels30_threshold': 0.9 * clv30, 'parsed_routing': parsed,
                      'server_model_path': info.get('model_path'), 'tracebacks_in_kernel_log': text.count('Traceback (most recent call last)'),
                      'serve_log_errors': len(re.findall(r'\bERROR\b|CUDA error', serve))}


def main():
    controls = {name: arm(out) for name, out in CONTROL.items()}
    (HERE / 'control-readout.json').write_text(json.dumps(controls, indent=2) + '\n')
    brief = lambda a: {'bench_clock': a['levels_benchmark_clock'], 'transcript_clock': a['levels_transcript_clock'],
                       'serve': a['serve_5_30_benchmark_clock'], 'states': a['state_counts']}
    if '--controls-only' in sys.argv:
        print(json.dumps({k: brief(v) for k, v in controls.items()}, indent=2))
        return
    if '--dry-run' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--dry-run') + 1])
        t, gates, metrics = evaluate(out, controls)
        print(json.dumps({'dry_run_on': str(out), 'arm': brief(t), 'gates': gates, 'metrics': metrics}, indent=2))
        return
    collection = json.loads((HERE / 'collection.json').read_text())
    result = {'candidate': 'A5B variant B: Intel W4A16 incumbent, num_experts_per_tok 10->5 (unhealed)',
              'kernel': collection['kernel'], 'version': collection['version'], 'status': 'INVALID_NO_QUEUE',
              'smoke_pass': False, 'queued': False, 'collection': {k: v for k, v in collection.items() if k != 'downloaded'}}
    if collection['status'] != 'COMPLETE':
        result['reason'] = 'Kernel did not complete; inspect exact-version logs.'
    else:
        t, gates, metrics = evaluate(HERE / 'output', controls)
        result.update(status='SMOKE_PASS' if all(gates.values()) else 'SMOKE_FAIL', smoke_pass=all(gates.values()),
                      gates=gates, metrics=metrics, a5b=t, controls={k: brief(v) for k, v in controls.items()})
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('a5b', 'controls')}, indent=2))


if __name__ == '__main__':
    main()
