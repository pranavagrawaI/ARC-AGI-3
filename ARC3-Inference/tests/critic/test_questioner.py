from __future__ import annotations

from types import SimpleNamespace

from inference.critic.policy import CriticModelConfig
from inference.critic.questioner import QuestionLedger, parse_question, similarity, validate_question
from inference.critic.questioner_runner import QuestionerRunner
from inference.critic.report import CriticReport, render_report_block


def test_parse_question_line():
    assert parse_question("question: Why must the piece move at all?") == "Why must the piece move at all?"
    assert parse_question("**Question:** Why?") == "Why?"
    assert parse_question("no label here") == ""


def test_validate_rejects_advice_and_accepts_questions():
    ledger = QuestionLedger()
    assert validate_question("Why must the piece move at all?", ledger) == ""
    assert validate_question("Click the bar at MOUSE(row=4, col=5)?", ledger) == "contains action syntax"
    assert validate_question("Try clicking the T bars?", ledger) == "is an instruction, not a question"
    assert validate_question("The goal is to fill the cup.", ledger) == "not a question"
    assert validate_question("", ledger) == "no `question:` line"


def test_validate_rejects_repeat_of_asked_question():
    ledger = QuestionLedger()
    ledger.add(1, 10, "Why does the blue block always stop four rows short of what it approaches?")
    assert similarity("Why does the block always stop four rows short?", ledger.entries[0].question) >= 0.5
    assert validate_question("Why does the blue block always stop four rows short?", ledger).startswith("repeats")
    assert validate_question("What have you never clicked on this board?", ledger) == ""


def test_reply_is_attached_to_pending_question():
    ledger = QuestionLedger()
    ledger.add(1, 10, "Why must the piece move at all?", after_seq=4)
    ledger.record_reply([
        {"seq": 4, "action_num": 10, "said": "Answer to observer: stale, same action number", "thinking": ""},
        {"seq": 5, "action_num": 10, "said": "World model: ...\nAnswer to observer: maybe the T states are the goal", "thinking": ""},
    ])
    assert ledger.entries[0].reply == "maybe the T states are the goal"


def test_question_style_renders_only_the_question():
    block = render_report_block(CriticReport(packet_id="q", summary="", open_questions=("Why?",), style="question"))
    text = "\n".join(block)
    assert "Why?" in text and "Recommended" not in text and "Summary" not in text
    assert render_report_block(CriticReport(packet_id="q", summary="", style="question")) == []


def _runner(replies: list[str]) -> tuple[QuestionerRunner, list[dict]]:
    sent: list[dict] = []

    def post(payload, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": replies[len(sent) - 1]}}], "usage": {"completion_tokens": 5}}

    model = CriticModelConfig(provider="vllm", base_url="http://x", model_id="m")
    return QuestionerRunner(model=model, period_turns=2, post=post), sent


def test_cadence_waits_period_turns_and_resets_on_level_change():
    runner, _ = _runner([])
    frame, ctx = SimpleNamespace(level=1), {"recent_words": []}
    fired = []
    for i in range(7):
        decision = runner.decide(history=[], level=1, action_num=i)
        fired.append(decision.fired)
        if decision.fired:
            runner.post = lambda payload, timeout: {"choices": [{"message": {"content": f"question: Why {i}?"}}]}
            runner.fire(decision, current_frame=frame, history=[], valid_actions=[], actor_context=ctx)
    # period 2: first question after 2 quiet turns in the level, then every 2nd turn
    assert fired == [False, False, True, False, True, False, True]
    assert runner.decide(history=[], level=2, action_num=8).fired is False


def test_fire_retries_once_on_rejection_and_passes_effort():
    runner, sent = _runner(["question: Try the bars?", "question: Why must the piece move at all?"])
    frame = SimpleNamespace(level=1)
    firing = runner.fire(
        SimpleNamespace(cause="period"), current_frame=frame, history=[], valid_actions=["UP"],
        actor_context={"recent_words": [{"action_num": 0, "said": "Plan: drop the piece in the cup", "thinking": "..."}]},
    )
    assert firing.parsed and firing.report.open_questions == ("Why must the piece move at all?",)
    assert len(sent) == 2 and sent[0]["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert "drop the piece in the cup" in sent[0]["messages"][1]["content"]
    assert runner.ledger.entries[0].question == "Why must the piece move at all?"


def test_fire_delivers_nothing_after_two_rejections():
    runner, _ = _runner(["question: Try X?", "Just do Y."])
    firing = runner.fire(SimpleNamespace(cause="period"), current_frame=SimpleNamespace(level=1), history=[],
                         valid_actions=[], actor_context={})
    assert not firing.parsed and render_report_block(firing.report) == []


def test_form_rotates_with_ledger():
    from inference.critic.questioner import QUESTION_FORMS, render_questioner_input
    ledger = QuestionLedger()
    seen = []
    for i in range(len(QUESTION_FORMS)):
        text = render_questioner_input(level=1, action_num=i, turns_since_level=i, recent_words=[], transitions=[], ledger=ledger)
        seen.append(next(l for l in text.splitlines() if l.startswith("Form for this question")))
        ledger.add(1, i, f"Why {i} thing{i}?")
    assert len(set(seen)) == len(QUESTION_FORMS)
