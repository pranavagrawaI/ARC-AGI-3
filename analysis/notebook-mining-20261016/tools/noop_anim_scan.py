"""Count no-op-guard batch stops and stale-state blocks that coincide with an 'animated ... not a no-op' note.
usage: python3 -I -B noop_anim_scan.py RUN_OUTPUT_DIR...  (dedups user/tool messages across requests)"""
import json, sys, glob, os, re
for d in sys.argv[1:]:
    seen = set(); c = dict(user_msgs=0, batch_stop=0, batch_stop_with_anim_note=0, anim_not_noop_note=0, stale_block=0, noop_echo=0)
    games = {}
    for f in sorted(glob.glob(os.path.join(d, '*_requests.jsonl'))):
        g = os.path.basename(f).split('_')[0]
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            for m in r.get('messages') or []:
                if m.get('role') not in ('user', 'tool'): continue
                x = m['content']
                s = x if isinstance(x, str) else ' '.join(y.get('text', '') for y in x if isinstance(y, dict))
                if s in seen: continue
                seen.add(s)
                if m['role'] == 'user':
                    c['user_msgs'] += 1
                    bs = 'stopped by the no-op guard' in s
                    an = 'but it was not a no-op' in s
                    c['batch_stop'] += bs; c['anim_not_noop_note'] += an
                    if bs and an:
                        c['batch_stop_with_anim_note'] += 1; games[g] = games.get(g, 0) + 1
                else:
                    c['stale_block'] += s.count('StaleStateActionError: ')
                    c['noop_echo'] += s.count('-> NO-OP (executed')
    print(os.path.relpath(d), json.dumps(c), games)
