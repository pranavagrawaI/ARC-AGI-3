"""Extract new turns only, preserving reasoning, code and matching tool receipts."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'transcript-audit'
OUT.mkdir(exist_ok=True)
all_rows = []
for directory in sorted((HERE / 'output/continuations/episodes').iterdir()):
    if not directory.is_dir():
        continue
    events = [json.loads(line) for line in (directory / 'wire.jsonl').read_text().splitlines()]
    tools = {}
    for event in events:
        if event['event'] == 'request':
            for message in event['body']['messages']:
                if message['role'] == 'tool':
                    tools[message['tool_call_id']] = message['content']
    turns = []
    text = []
    for event in events:
        if event['event'] != 'response':
            continue
        body = event['body']; choice = body['choices'][0]; message = choice['message']
        turn = {'call': event['call'], 'tokens': body['usage']['completion_tokens'],
                'finish_reason': choice['finish_reason'], 'latency_seconds': event['latency_seconds'],
                'reasoning': message.get('reasoning_content') or '', 'content': message.get('content'), 'tools': []}
        text.append(f"\n=== CALL {turn['call']} | {turn['tokens']} tokens | {turn['finish_reason']} ===\n")
        text.append(turn['reasoning'])
        for tool in message.get('tool_calls') or []:
            arguments = tool['function']['arguments']
            try:
                code = json.loads(arguments).get('code', arguments)
            except (ValueError, AttributeError):
                code = arguments
            # Display literal newline escapes as source lines; raw arguments remain preserved.
            display = code.replace('\\n', '\n')
            receipt = tools.get(tool['id'])
            try:
                parsed = json.loads(receipt) if receipt else None
            except (ValueError, TypeError):
                parsed = receipt
            item = {'id': tool['id'], 'name': tool['function']['name'], 'arguments': arguments,
                    'display_code': display, 'receipt': parsed}
            turn['tools'].append(item)
            text.append('\nCODE\n'+display+'\nRESULT\n'+json.dumps(parsed, indent=2, ensure_ascii=False))
        turns.append(turn)
    (OUT / (directory.name+'.json')).write_text(json.dumps(turns, indent=2, ensure_ascii=False)+'\n')
    (OUT / (directory.name+'.txt')).write_text('\n'.join(text)+'\n')
    result = json.loads((directory / 'result.json').read_text())
    prefix = json.loads((directory / 'checkpoint.json').read_text())['prefix_actions']
    all_rows.append({'episode': directory.name, 'calls': len(turns), 'tokens': sum(t['tokens'] for t in turns),
        'no_tool_tokens': sum(t['tokens'] for t in turns if not t['tools']),
        'length_tokens': sum(t['tokens'] for t in turns if t['finish_reason']=='length'),
        'tool_calls': sum(len(t['tools']) for t in turns),
        'tool_errors': sum(bool(isinstance(x['receipt'],dict) and x['receipt'].get('error')) for t in turns for x in t['tools']),
        'uncollected_tool_receipts': sum(x['receipt'] is None for t in turns for x in t['tools']),
        'new_actions': result['total_actions']-len(prefix)})
(OUT / 'index.json').write_text(json.dumps(all_rows, indent=2)+'\n')
print(json.dumps(all_rows, indent=2))
