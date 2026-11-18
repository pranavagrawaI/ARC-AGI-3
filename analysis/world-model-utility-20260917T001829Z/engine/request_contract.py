from protocol import project_messages
from common import CAPS,MODEL

def validate_request(body,phase='constructor'):
 if set(body)!={'model','messages','provider','tools','tool_choice','max_tokens','reasoning'}:raise ValueError('logical fields')
 if body['model']!=MODEL or body['provider']!={} or body['reasoning']!={'enabled':False}:raise ValueError('logical identity')
 if type(body['max_tokens']) is not int or not 1<=body['max_tokens']<=CAPS[phase]['completion_per_request'] or body['tool_choice']!='required':raise ValueError('request bounds')
 if project_messages(body['messages'])!=body['messages']:raise ValueError('unprojected messages')
 names={t['function']['name'] for t in body['tools']}
 allowed={'execute_python','submit_candidate','request_evidence'} if phase=='constructor' else {'act','retrieve','query_model'}
 if not names or not names<=allowed:raise ValueError('tools')
