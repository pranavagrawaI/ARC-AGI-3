"""Weight context regimes by recorded Actor request wall time, without inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ('analysis/deployment-probes/20260906-q6-summary-boundary/raw/'
                 'nooa-join-public3/solver-workspaces/ls20-8md_11v1/ipc/model_calls.jsonl')


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    actor = [row for row in rows if not row['summarizer']]
    total = sum(row['wall_seconds'] for row in actor)
    bins = []
    for lower, upper in [(0, 10_000), (10_000, 22_000), (22_000, 64_000),
                         (64_000, 95_000), (95_000, None)]:
        selected = [row for row in actor if row['usage']['prompt_tokens'] >= lower
                    and (upper is None or row['usage']['prompt_tokens'] < upper)]
        wall = sum(row['wall_seconds'] for row in selected)
        bins.append({'lower_inclusive': lower, 'upper_exclusive': upper,
                     'requests': len(selected), 'summed_request_wall_s': wall,
                     'fraction_of_summed_actor_request_wall': wall / total})
    result = {'source': str(SOURCE.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'actor_requests': len(actor), 'summed_actor_request_wall_s': total,
              'context_bins': bins,
              'limits': 'One real game run. Request wall includes multiple stages; this is workload weighting, not context causality, pure GPU time, or a cache measurement.'}
    Path(__file__).with_name('PERFORMANCE-WORKLOAD.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
