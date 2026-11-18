"""Separate fail-closed contracts for logical OpenRouter and local SGLang wires."""
import copy
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGICAL_MODEL = 'qwen/qwen3.8-27b'
LOCAL_MODEL = '/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot'
LOGICAL_ENVELOPE = {'model', 'messages', 'provider'}
LOGICAL_ESSENTIAL = {'tools', 'tool_choice', 'max_tokens', 'reasoning'}
LOCAL_FIELDS = {'model', 'messages', 'tools', 'tool_choice', 'max_tokens', 'reasoning', 'chat_template_kwargs'}


def _require(condition, message):
    if not condition:
        raise ValueError('request contract: ' + message)


def _validate_messages_and_tools(body):
    tools = body['tools']
    _require(isinstance(tools, list) and bool(tools), 'tools required')
    names = []
    for tool in tools:
        _require(set(tool) == {'type', 'function'} and tool['type'] == 'function', 'function tool envelope')
        function = tool['function']; names.append(function['name'])
        _require(set(function) == {'name', 'description', 'parameters'}, 'unsupported function options')
        _require(isinstance(function['description'], str), 'tool description')
        schema = function['parameters']
        _require(set(schema) == {'type', 'properties', 'required', 'additionalProperties'} and schema['type'] == 'object' and schema['additionalProperties'] is False, 'tool object schema')
        _require(set(schema['required']) <= set(schema['properties']), 'required properties')
        _require(all(set(value) == {'type'} and value['type'] in ('string', 'object') for value in schema['properties'].values()), 'property schema')
    _require(len(names) == len(set(names)) and set(names) <= {'execute_python', 'submit_candidate', 'request_evidence'} and 'request_evidence' in names, 'allowed tools')
    messages = body['messages']
    _require(isinstance(messages, list) and bool(messages), 'messages required')
    pending = set()
    for message in messages:
        _require(isinstance(message, dict), 'message object')
        role = message.get('role')
        _require(role in ('system', 'user', 'assistant', 'tool'), 'message role')
        allowed = {'role', 'content'} | ({'tool_calls'} if role == 'assistant' else {'tool_call_id'} if role == 'tool' else set())
        _require(set(message) <= allowed, 'unreviewed message extension')
        _require(isinstance(message.get('content'), str) or (role == 'assistant' and message.get('content') is None), 'text-only message content')
        if role == 'assistant':
            _require(not pending, 'unanswered previous tools')
            for call in message.get('tool_calls') or []:
                _require(set(call) == {'id', 'type', 'function'} and call['type'] == 'function' and isinstance(call['id'], str) and call['id'] and call['id'] not in pending, 'tool call envelope')
                _require(set(call['function']) == {'name', 'arguments'} and isinstance(call['function']['arguments'], str), 'tool call function')
                pending.add(call['id'])
        elif role == 'tool':
            _require(message.get('tool_call_id') in pending, 'unmatched tool result')
            pending.remove(message['tool_call_id'])
        else:
            _require(not pending, 'unanswered tool calls before message')
    _require(not pending, 'unanswered tool calls')


def validate_request(body, selection=None, model=None):
    """Validate the constructor's logical OpenRouter request without implying its wire."""
    selection = selection or json.loads((HERE/'PROVIDER-SELECTION.json').read_text())
    model = model or json.loads((HERE/'model-metadata.json').read_text())
    endpoint = selection['selected']
    _require(isinstance(body, dict), 'body must be object')
    optional = set(body) - LOGICAL_ENVELOPE
    _require(optional <= set(endpoint['supported_parameters']), 'unsupported optional fields: ' + str(sorted(optional-set(endpoint['supported_parameters']))))
    _require(set(body) == LOGICAL_ENVELOPE | LOGICAL_ESSENTIAL, 'unexpected or missing fields in logical request')
    _require(body['model'] == LOGICAL_MODEL == endpoint['model_id'] == model['id'], 'exact logical model identity')
    _require(endpoint['status'] == 0, 'endpoint not advertised active')
    expected = {'only':[endpoint['tag']], 'order':[endpoint['tag']], 'allow_fallbacks':False, 'require_parameters':True, 'max_price':{'prompt':1,'completion':10}}
    _require(body['provider'] == selection['routing'] == expected, 'routing drift or fallback')
    _require(Decimal(endpoint['pricing']['prompt']) <= Decimal('0.000001') and Decimal(endpoint['pricing']['completion']) <= Decimal('0.00001'), 'advertised prices exceed cap')
    _require(body['reasoning'] == {'enabled':False}, 'reasoning must be disabled')
    _require(model.get('reasoning', {}).get('mandatory') is False, 'disabling reasoning not established')
    _require(body['tool_choice'] == 'required' and endpoint['supports_tool_choice'].get('required') is True, 'required tool choice unsupported')
    _require(type(body['max_tokens']) is int and 1 <= body['max_tokens'] <= min(4096, endpoint['max_completion_tokens']), 'completion allowance')
    reservation = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    _require(reservation + body['max_tokens'] <= endpoint['context_length'], 'context allowance')
    _require(not endpoint.get('max_prompt_tokens') or reservation <= endpoint['max_prompt_tokens'], 'prompt allowance')
    _validate_messages_and_tools(body)
    return {'passed': True, 'wire': 'logical-openrouter', 'prompt_reservation': reservation, 'actual_provider_service': 'NOT_ESTABLISHED_BY_PREFLIGHT'}


def logical_to_local_request(body, *, local_model):
    """Mechanically adapt a validated logical request to the proven v15 local wire."""
    validate_request(body)
    _require(local_model == LOCAL_MODEL, 'local model identity drift')
    local = {key: copy.deepcopy(body[key]) for key in ('messages', 'tools', 'tool_choice', 'max_tokens', 'reasoning')}
    local['model'] = local_model
    local['chat_template_kwargs'] = {'enable_thinking': False}
    validate_local_request(local, expected_model=local_model)
    return local


def validate_local_request(body, *, expected_model=LOCAL_MODEL):
    """Validate only fields supported by the v15 SGLang/vLLM Chat Completions wire."""
    _require(isinstance(body, dict), 'local body must be object')
    _require(set(body) == LOCAL_FIELDS, 'unexpected or missing local wire fields')
    _require('provider' not in body, 'gateway provider field on local wire')
    _require(expected_model == LOCAL_MODEL and body['model'] == expected_model, 'local model identity drift')
    _require(body['reasoning'] == {'enabled': False}, 'local reasoning control missing')
    _require(body['chat_template_kwargs'] == {'enable_thinking': False}, 'local thinking control missing')
    _require(body['tool_choice'] == 'required', 'local required tool choice')
    _require(type(body['max_tokens']) is int and 1 <= body['max_tokens'] <= 4096, 'local completion allowance')
    _validate_messages_and_tools(body)
    return {'passed': True, 'wire': 'local-sglang-vllm', 'model': body['model']}
