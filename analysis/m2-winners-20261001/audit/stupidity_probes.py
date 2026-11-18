"""CPU-only probes of existing KF/P plumbing; no model, game, or launch."""
import ast
import base64
import collections
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib

HERE = Path(__file__).resolve().parent
FLASH = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next')
SOURCE = FLASH / 'kf/source-dataset/src/ARC3-Inference/inference/agent/tool_agent.py'


def main():
    tree = ast.parse(SOURCE.read_text())
    names = {'_estimate_request_input_tokens', '_drop_oldest_history_block',
             '_keep_recent_history_turns', '_drop_until_first_user_message',
             '_persistent_history_messages', '_trim_messages_for_context'}
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ToolAgent')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    estimate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_estimate_tokens')
    cls = ast.ClassDef(name='Probe', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), estimate, cls], type_ignores=[])
    ns = {'json': json, '_PERSISTENT_HISTORY_ASSISTANT_TURNS': 30}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
    agent = ns['Probe']()
    agent._context_budget_tokens = 31744
    result = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'scope': 'Exact extracted history/estimate methods, synthetic fixtures, no server or gameplay.'}

    retention = []
    for count in (29, 30, 31):
        messages = [{'role': 'system', 'content': 'Solve the game.'}, {'role': 'user', 'content': 'Observed board.'}]
        for i in range(count):
            messages += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(i), 'type': 'function', 'function': {'name': 'python', 'arguments': '{"code":"print(1)"}'}}]},
                         {'role': 'tool', 'tool_call_id': str(i), 'content': '1'}]
        kept = agent._persistent_history_messages(messages)
        retention.append({'assistant_messages': count, 'estimated_request_tokens': agent._estimate_request_input_tokens(messages),
                          'history_messages_before': len(messages)-1, 'history_messages_after': len(kept)})
    assert [x['history_messages_after'] for x in retention] == [59, 61, 0]
    result['history_boundary'] = retention

    donor_path = HERE / 'lord-bundle/src/ARC3-Inference/inference/agent/tool_agent.py'
    donor_tree = ast.parse(donor_path.read_text())
    extend = next(n for n in ast.walk(donor_tree) if isinstance(n, ast.FunctionDef) and n.name == '_extend_to_turn_start')
    extend.decorator_list = []
    helper = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), extend], type_ignores=[])
    exec(compile(ast.fix_missing_locations(helper), str(donor_path), 'exec'), ns)
    hist = messages[1:]
    cut = agent._keep_recent_history_turns(hist, max_turns=30)
    restored = ns['_extend_to_turn_start'](agent, hist, cut)
    assert restored == hist
    result['donor_boundary_repair_probe'] = {'source':str(donor_path), 'source_sha256':hashlib.sha256(donor_path.read_bytes()).hexdigest(),
        'restored_history_messages':len(restored), 'equals_original_history':restored == hist,
        'scope':'Calls donor helper on the exact KF cutoff result. Not installed in KF/P; no over-budget or gameplay qualification.'}

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag+data))
    pixels = b''.join(b'\0' + bytes([row % 16])*256 for row in range(256))
    images = []
    for level in (0, 9):
        compressed = zlib.compress(pixels, level)
        assert zlib.decompress(compressed) == pixels
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB',256,256,8,0,0,0,0)) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')
        msg = [{'role': 'user', 'content': [{'type':'image_url', 'image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]}]
        images.append({'png_compression_level':level, 'png_bytes':len(png), 'estimated_request_tokens':agent._estimate_request_input_tokens(msg)})
    result['same_pixels_different_estimates'] = {'size':[256,256], 'raw_pixels_sha256':hashlib.sha256(pixels).hexdigest(), 'variants':images,
        'interpretation':'Identical decoded pixels and dimensions; estimate varies with transport compression. No actual model-token count measured.'}

    raw = Path('analysis/speed-screen-resume-20260930/screen-out/bench/raw-S0.json')
    records = json.loads(raw.read_text())['records']
    errors = [r for r in records if r.get('status') == 400]
    totals = collections.Counter()
    for r in errors:
        match = re.search(r'total of at least (\d+) tokens', r['error'])
        if match:
            totals[int(match[1])] += 1
    result['archived_speed_screen_S0'] = {'source':str(raw), 'status_counts':dict(collections.Counter(r['status'] for r in records)),
        'context_errors':sum('maximum context length' in r.get('error','') for r in errors),
        'reported_lower_bounds_on_total':dict(totals),
        'qualification':'Messages say at least: this does not establish an exact one-token true overflow. Revised benchmark code exists separately; no repair or rerun here.'}

    result['P_history_status_census'] = {}
    for arm in ('out-p1', 'out-p2'):
        count = empty = 0
        for file in (FLASH / arm / 'artifacts').glob('*events.jsonl'):
            for line in file.open():
                e = json.loads(line)
                if e.get('type') != 'analysis': continue
                found = re.findall(r'^history_messages: (\d+)', e.get('transcript',''), re.M)
                if found:
                    count += 1
                    empty += int(found[-1]) == 0
        result['P_history_status_census'][arm] = {'analysis_events_with_history_status':count, 'reported_empty_history':empty,
            'limit':'Status location/timing was not qualified as an exact persistence-event monitor; absence is not a proof the defect never triggered.'}

    (HERE / 'stupidity-probes.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
