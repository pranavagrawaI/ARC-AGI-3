"""Fail-closed contract for this frozen text-only Chat Completions episode.

OpenRouter's unified reasoning object is a gateway control, not a promise that
an upstream API accepts a field with that spelling. Endpoint metadata advertises
`reasoning`; model metadata separately establishes disabling is permitted.
"""
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = 'qwen/qwen3.8-27b'
ENVELOPE = {'model', 'messages', 'provider'}
ESSENTIAL = {'tools', 'tool_choice', 'max_tokens', 'reasoning'}


def validate_request(body, selection=None, model=None):
    selection = selection or json.loads((HERE/'PROVIDER-SELECTION.json').read_text())
    model = model or json.loads((HERE/'model-metadata.json').read_text())
    endpoint = selection['selected']
    def require(condition, message):
        if not condition:
            raise ValueError('request contract: ' + message)
    require(isinstance(body, dict), 'body must be object')
    optional = set(body) - ENVELOPE
    require(optional <= set(endpoint['supported_parameters']), 'unsupported optional fields: ' + str(sorted(optional-set(endpoint['supported_parameters']))))
    require(set(body) == ENVELOPE | ESSENTIAL, 'unexpected or missing fields in minimal frozen request')
    require(body['model'] == MODEL == endpoint['model_id'] == model['id'], 'exact model identity')
    require(endpoint['status'] == 0, 'endpoint not advertised active')
    expected_routing = {'only':[endpoint['tag']], 'order':[endpoint['tag']], 'allow_fallbacks':False,
                        'require_parameters':True, 'max_price':{'prompt':1,'completion':10}}
    require(body['provider'] == selection['routing'] == expected_routing, 'routing drift or fallback')
    require(Decimal(endpoint['pricing']['prompt']) <= Decimal('0.000001') and
            Decimal(endpoint['pricing']['completion']) <= Decimal('0.00001'), 'advertised prices exceed cap')
    require(body['reasoning'] == {'enabled':False}, 'reasoning must be disabled, not merely hidden')
    require(model.get('reasoning', {}).get('mandatory') is False, 'disabling reasoning not established')
    require(body['tool_choice'] == 'required' and endpoint['supports_tool_choice'].get('required') is True, 'required tool choice unsupported')
    require(type(body['max_tokens']) is int and 1 <= body['max_tokens'] <= min(4096,endpoint['max_completion_tokens']), 'completion allowance')
    prompt_reservation = len(json.dumps(body,ensure_ascii=False).encode()) + 4096
    require(prompt_reservation + body['max_tokens'] <= endpoint['context_length'], 'context allowance')
    require(not endpoint.get('max_prompt_tokens') or prompt_reservation <= endpoint['max_prompt_tokens'], 'prompt allowance')
    tools = body['tools']
    require(isinstance(tools,list) and bool(tools), 'tools required')
    names=[]
    for tool in tools:
        require(set(tool)=={'type','function'} and tool['type']=='function', 'function tool envelope')
        f=tool['function']; names.append(f['name'])
        require(set(f)=={'name','description','parameters'}, 'unsupported function options (strict is not requested)')
        require(isinstance(f['description'],str), 'tool description')
        schema=f['parameters']
        require(set(schema)=={'type','properties','required','additionalProperties'} and schema['type']=='object' and schema['additionalProperties'] is False, 'tool object schema')
        require(set(schema['required']) <= set(schema['properties']), 'required properties')
        require(all(set(v)=={'type'} and v['type'] in ('string','object') for v in schema['properties'].values()), 'property schema')
    require(len(names)==len(set(names)) and set(names)<={'execute_python','submit_candidate','request_evidence'} and 'request_evidence' in names, 'allowed tools')
    messages=body['messages']
    require(isinstance(messages,list) and bool(messages), 'messages required')
    pending=set()
    for message in messages:
        role=message.get('role')
        require(role in ('system','user','assistant','tool'), 'message role')
        allowed={'role','content'} | ({'tool_calls','reasoning','reasoning_details','refusal','annotations'} if role=='assistant' else {'tool_call_id'} if role=='tool' else set())
        require(set(message)<=allowed, 'unreviewed message extension')
        require(isinstance(message.get('content'),str) or (role=='assistant' and message.get('content') is None), 'text-only message content')
        if role=='assistant':
            require(not message.get('reasoning') and not message.get('reasoning_details'), 'unexpected reasoning in transcript')
            require(not pending, 'unanswered previous tools')
            for call in message.get('tool_calls') or []:
                require(set(call)=={'id','type','function'} and call['type']=='function' and isinstance(call['id'],str) and call['id'] not in pending, 'tool call envelope')
                require(set(call['function'])=={'name','arguments'} and isinstance(call['function']['arguments'],str), 'tool call function')
                pending.add(call['id'])
        elif role=='tool':
            require(message.get('tool_call_id') in pending, 'unmatched tool result')
            pending.remove(message['tool_call_id'])
        else:
            require(not pending, 'unanswered tool calls before message')
    require(not pending, 'unanswered tool calls')
    return {'passed':True,'envelope_fields':sorted(ENVELOPE),'optional_fields':sorted(optional),
            'unsupported_optional_fields':[], 'prompt_reservation':prompt_reservation,
            'context_length':endpoint['context_length'], 'reasoning_control':'gateway enabled=false; model mandatory=false; no effort-name equivalence assumed',
            'actual_provider_service':'NOT_ESTABLISHED_BY_PREFLIGHT'}
