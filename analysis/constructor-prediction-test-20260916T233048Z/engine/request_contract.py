"""Constructor logical relay contract; no gateway routing or billing."""
from protocol import project_messages

def validate_request(body):
    if set(body) != {'model', 'messages', 'provider', 'tools', 'tool_choice', 'max_tokens', 'reasoning'}:
        raise ValueError('constructor fields')
    if body['model'] != 'qwen/qwen3.8-27b' or body['provider'] != {} or body['reasoning'] != {'enabled': False}:
        raise ValueError('constructor logical identity/control')
    if type(body['max_tokens']) is not int or not 1 <= body['max_tokens'] <= 4096 or body['tool_choice'] != 'required':
        raise ValueError('constructor bound')
    if project_messages(body['messages']) != body['messages']:
        raise ValueError('unprojected constructor messages')
    names = {t['function']['name'] for t in body['tools']}
    if not names <= {'execute_python', 'submit_candidate', 'request_evidence'} or 'request_evidence' not in names:
        raise ValueError('constructor tools')
    return {'passed': True, 'wire': 'logical-local-relay'}
