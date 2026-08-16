from guided_first_use import GUIDED_FIRST_USE_SCHEMA, render_guided_first_use


def test_repeated_manual_reconstruction_gets_one_exact_read_instruction() -> None:
    rendered = render_guided_first_use(
        condition_codes=("evidence_without_commitment", "repeated_action_evidence"),
        current_model_version_id=None,
        world_model_was_read=False,
    )

    assert GUIDED_FIRST_USE_SCHEMA == "0159-guided-first-use-v1"
    assert rendered is not None
    assert len(rendered.encode("utf-8")) <= 1_536
    assert "self.read_world_model()" in rendered
    assert "read-only" in rendered
    assert "does not create a commitment" in rendered
    assert "narrow executable rule" in rendered
    assert "prose notes" in rendered
    assert "defer" in rendered
    assert "You decide the hypothesis" in rendered


def test_guidance_is_absent_before_repetition_after_read_or_with_a_model() -> None:
    cases = (
        {
            "condition_codes": ("evidence_without_commitment",),
            "current_model_version_id": None,
            "world_model_was_read": False,
        },
        {
            "condition_codes": ("repeated_action_evidence",),
            "current_model_version_id": None,
            "world_model_was_read": True,
        },
        {
            "condition_codes": ("repeated_action_evidence",),
            "current_model_version_id": "model-version-00000000",
            "world_model_was_read": False,
        },
    )

    assert all(render_guided_first_use(**case) is None for case in cases)
