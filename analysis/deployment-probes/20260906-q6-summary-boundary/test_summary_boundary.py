"""Regression at the real NOOA compression boundary; no external model calls."""
import asyncio
import ast
import copy
import json
from pathlib import Path

import pytest
from test_payload import MdArcSolverAgent, FakeLLMClient, LLMResponse, response, rows, ROOT

OLD = ROOT / 'analysis/deployment-probes/20260906-q6-prediction-feedback'
NOTE = "Observed: UP moved 5. Hypothesis: LEFT blocked? Untested.\nquotes: ' \" \\\nUnicode: λ. history_markdown = 'decoy'\ntarget_chars = 12\n"


def summary_response(text):
    content = json.dumps(text)
    return LLMResponse(raw_response=None, content=content, tool_calls=[], finish_reason='stop',
                       assistant_message={'role': 'assistant', 'content': content})


def state(run, turn=0):
    record = {'turn': turn, 'step': turn, 'grid_rows': ['00', '00'],
              'state': 'NOT_FINISHED', 'level': 0, 'levels_completed': 0,
              'available_actions': ['UP'], 'action_results': [], 'note': ''}
    with (run / 'ipc/states.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')


async def actor_with_note(tmp_path):
    first = response(f"self.write_notes('facts.md', {NOTE!r})\nprint(self.read_notes('facts.md'))\nself.submit_actions(['UP'], rationale='predict: move')")
    second = response("self.submit_actions(['UP'], rationale='predict: move')")
    second.tool_calls[0].id = 'two'
    fake = FakeLLMClient(scripted_responses=[first, summary_response('Unresolved LEFT hypothesis; recorded call one.'), second])
    actor = MdArcSolverAgent(llm=fake, run_dir=tmp_path)
    await asyncio.sleep(0)
    state(tmp_path)
    await actor.handle({'user_messages': ['Read notes and make one move.']})
    assert actor._integrity.outputs
    return actor, fake


@pytest.mark.parametrize('extra', ['', "\nExample '''quoted'''", '\nExample """quoted"""', "\n\n### Observations\nBoth ''' and \"\"\" quotes", '\n\n### User Message `{"tag": "123"}`\n\nAn authored example.'])
def test_real_note_compression_then_actor_resume(tmp_path, monkeypatch, extra):
    monkeypatch.setitem(globals(), 'NOTE', NOTE + extra)
    tmp_path = tmp_path / 'session'
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    async def run():
        actor, fake = await actor_with_note(tmp_path)
        try:
            summarizer = actor._summarizers[0]
            tags = actor.event_manager.keys()
            summarizer._schedule_summarization(tags[0], tags[-1])
            await summarizer._pending_task
            assert summarizer._pending_summary is not None
            state(tmp_path, 1)
            # Actual BeforeTurn applies the completed summary, then Actor calls LLM.
            await actor.handle({'user_messages': ['Continue with one move.']})
            assert any(type(e).__name__ == 'Summary' for e in actor.event_manager.values())
            assert [r['summarizer'] for r in rows(tmp_path / 'ipc/model_calls.jsonl')] == [False, True, False]
            assert fake.call_count == 3
            assert not (tmp_path / 'ipc/ABORT').exists()
            evidence = rows(tmp_path / 'ipc/integrity.jsonl')
            assert any(r['event'] == 'SUMMARY_INPUT_VERIFIED' for r in evidence)
            assert (actor.knowledge_dir / 'facts.md').read_text() == NOTE
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


@pytest.mark.parametrize('boundary', ['history', 'request', 'request_with_decoy', 'request_with_separate_decoy'])
def test_corruption_is_rejected_before_model(tmp_path, monkeypatch, boundary):
    tmp_path = tmp_path / 'session'
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    async def run():
        actor, fake = await actor_with_note(tmp_path)
        try:
            summarizer = actor._summarizers[0]
            tags = actor.event_manager.keys()
            history = summarizer._render_range_to_markdown(tags[0], tags[-1])
            if boundary == 'history':
                history = history.replace('Observed: UP moved 5.', 'Observed: UP moved 9.')
            else:
                # Mutate at the last request boundary, immediately before validation.
                original = actor._integrity.check_summary_messages
                def corrupt(messages, summarizer):
                    changed = copy.deepcopy(messages)
                    for m in changed:
                        if m['role'] == 'user' and '## Input parameters:' in m.get('content', ''):
                            good = m['content']
                            m['content'] = good.replace('Observed: UP moved 5.', 'Observed: UP moved 9.')
                            if boundary == 'request_with_decoy':
                                m['content'] += '\nIntact decoy: ' + good
                    if boundary == 'request_with_separate_decoy':
                        changed.append({'role': 'user', 'content': 'Summary containing intact copy: ' + good})
                    return original(changed, summarizer)
                monkeypatch.setattr(actor._integrity, 'check_summary_messages', corrupt)
            with pytest.raises(RuntimeError, match='differs'):
                await summarizer.summarize(history, target_chars=4000)
            assert fake.call_count == 1
            assert (tmp_path / 'ipc/ABORT').exists()
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


def test_saved_kaggle_request_through_real_summarizer(tmp_path, monkeypatch):
    tmp_path = tmp_path / 'session'
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    raw = OLD / 'long-run/raw/nooa-join-public3/solver-workspaces/ls20-w_gx27z4/ipc/integrity.jsonl'
    failure = next(r for r in reversed(rows(raw)) if r['event'] == 'INTEGRITY_FAILED')
    content = next(m['content'] for m in failure['messages'] if 'history_markdown = ' in m['content'])
    history = ast.literal_eval(content.split('history_markdown = ', 1)[1].rsplit('\ntarget_chars = ', 1)[0])
    async def run():
        fake = FakeLLMClient(scripted_responses=[summary_response('Captured history summary')])
        actor = MdArcSolverAgent(llm=fake, run_dir=tmp_path)
        await asyncio.sleep(0)
        try:
            # Exact captured note witness; no future game data supplied.
            guard = actor._integrity
            tid = 'call_a5e73576e435413fae5870ab'
            witness = next(r for r in rows(raw) if r['event'] == 'NOTE_RENDER' and r['tool_call_id'] == tid)
            guard.outputs[tid] = (failure['expected'], [{'name': witness['name'], 'stored': witness['stored']}])
            # Both the exact failed request and a newly rendered real request must pass.
            original = guard.check_summary_messages
            def check_captured_and_live(messages, summarizer):
                original(failure['messages'], summarizer)
                original(messages, summarizer)
            monkeypatch.setattr(guard, 'check_summary_messages', check_captured_and_live)
            summary = await actor._summarizers[0].summarize(history, target_chars=4000)
            assert summary == 'Captured history summary'
            assert fake.call_count == 1
            assert rows(tmp_path / 'ipc/model_calls.jsonl')[0]['summarizer']
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


def test_summary_retry_and_repeated_compression(tmp_path, monkeypatch):
    tmp_path = tmp_path / 'session'
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    async def run():
        actor, fake = await actor_with_note(tmp_path)
        try:
            fake._response_queue.clear()
            # Wrong JSON return type forces Predict to retry the same bound input.
            fake._response_queue.extend([summary_response([]), summary_response('Retain call one and unresolved LEFT test.')])
            summarizer = actor._summarizers[0]
            tags = actor.event_manager.keys()
            summarizer._schedule_summarization(tags[0], tags[-1])
            await summarizer._pending_task
            assert summarizer._pending_summary == 'Retain call one and unresolved LEFT test.'
            summarizer._apply_pending_summary()
            assert actor._integrity.summary_input.get() is None
            fake._response_queue.append(summary_response('Still unresolved; source call one archived.'))
            tags = actor.event_manager.keys()
            summarizer._schedule_summarization(tags[0], tags[-1])
            await summarizer._pending_task
            assert summarizer._pending_summary == 'Still unresolved; source call one archived.'
            summarizer._apply_pending_summary()
            assert [r['summarizer'] for r in rows(tmp_path / 'ipc/model_calls.jsonl')] == [False, True, True, True]
            assert len([r for r in rows(tmp_path / 'ipc/integrity.jsonl') if r['event'] == 'SUMMARY_INPUT_VERIFIED']) == 3
            assert not (tmp_path / 'ipc/ABORT').exists()
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


@pytest.mark.parametrize('decoy', [False, True])
def test_actor_retained_note_corruption_still_fails(tmp_path, monkeypatch, decoy):
    tmp_path = tmp_path / 'session'
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    async def run():
        actor, fake = await actor_with_note(tmp_path)
        try:
            original = actor._integrity.check_messages
            def corrupt(messages, active_output_ids=None):
                changed = copy.deepcopy(messages)
                for m in changed:
                    if m['role'] == 'user' and 'PythonOutput(' in m.get('content', ''):
                        m['content'] = m['content'].replace('Observed: UP moved 5.', 'Observed: UP moved 9.')
                if decoy:
                    expected = next(iter(actor._integrity.outputs.values()))[0]
                    changed.append({'role': 'user', 'content': '<sys tag="archived">\nSummary(content=' + expected + ')\n</sys>'})
                return original(changed, active_output_ids)
            monkeypatch.setattr(actor._integrity, 'check_messages', corrupt)
            state(tmp_path, 1)
            with pytest.raises(Exception, match='visible note event differs'):
                await actor.handle({'user_messages': ['Continue.']})
            assert fake.call_count == 1
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


def test_history_without_old_note_reports_unobserved(tmp_path, monkeypatch):
    tmp_path = tmp_path / 'session'
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    async def run():
        actor, fake = await actor_with_note(tmp_path)
        try:
            summary = await actor._summarizers[0].summarize('Only newer observations fit in this range.', 4000)
            assert summary
            evidence = rows(tmp_path / 'ipc/integrity.jsonl')
            assert any(r['event'] == 'NOTE_COVERAGE' and r['coverage'] == 'UNOBSERVED' for r in evidence)
            assert fake.call_count == 2
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())


def test_unbound_summarizer_request_cannot_skip_integrity(tmp_path):
    async def run():
        actor = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path / 'session')
        await asyncio.sleep(0)
        try:
            with pytest.raises(RuntimeError, match='no bound history input'):
                actor._integrity.check_summary_messages([], actor._summarizers[0])
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())
