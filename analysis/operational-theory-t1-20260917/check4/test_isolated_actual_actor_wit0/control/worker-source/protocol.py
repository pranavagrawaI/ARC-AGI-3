"""Shared OpenRouter protocol projection. Raw receipts are never projected."""
import copy
from local_wire import project_messages as original_projection

def assistant_message(message):
    result = {'role':'assistant', 'content':message.get('content')}
    if message.get('tool_calls'):
        result['tool_calls']=[{'id':c['id'],'type':c['type'],'function':{k:c['function'][k] for k in ('name','arguments')}} for c in message['tool_calls']]
    if message.get('reasoning_details'):
        result['reasoning_details']=copy.deepcopy(message['reasoning_details'])
    elif message.get('reasoning'):
        result['reasoning']=message['reasoning']
    elif message.get('reasoning_content'):
        raise ValueError('undocumented reasoning_content alias; no silent translation')
    return result

def project_messages(messages):
    stripped=copy.deepcopy(messages)
    for m in stripped:
        m.pop('reasoning_details',None);m.pop('reasoning',None)
    projected=original_projection(stripped)
    for original,m in zip(messages,projected):
        if original.get('role')=='assistant':
            for key in ('reasoning_details','reasoning'):
                if key in original: m[key]=copy.deepcopy(original[key])
    return projected
