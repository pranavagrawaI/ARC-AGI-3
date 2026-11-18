from openrouter_probe import request_purpose


def test_request_purpose_distinguishes_actor_from_summarizer():
    actor = {'tools': [{'type': 'function', 'function': {'name': 'execute_python'}}]}
    summarizer = {'messages': [{'role': 'user', 'content': 'summarize'}]}
    assert request_purpose(actor) == 'actor'
    assert request_purpose(summarizer) == 'summarizer'
