"""Shared Actor/critic request contract; treatment affects only Actor context."""
from protocol import project_messages


def validate_request(body):
    required = {'model', 'messages', 'max_tokens'}
    if not required <= body.keys() or set(body) - required - {'tools', 'tool_choice'}:
        raise ValueError('Actor request fields')
    if body['model'] != 'qwen/qwen3.8-27b':
        raise ValueError('Actor model identity')
    if type(body['max_tokens']) is not int or not 1 <= body['max_tokens'] <= 4096:
        raise ValueError('Actor completion bound')
    if project_messages(body['messages']) != body['messages']:
        raise ValueError('unprojected Actor messages')
    if 'tools' in body:
        if body.get('tool_choice') != 'auto' or not isinstance(body['tools'], list) or not body['tools']:
            raise ValueError('Actor tools')
        for tool in body['tools']:
            if tool.get('type') != 'function' or not isinstance(tool.get('function', {}).get('parameters'), dict):
                raise ValueError('Actor function schema')
    elif 'tool_choice' in body:
        raise ValueError('tool_choice without tools')
