from pathlib import Path

import run_live_ls20
from run_candidate_ls20 import CANDIDATES, configure


def test_candidate_routes_are_exact_and_non_fallback() -> None:
    deepseek = CANDIDATES["deepseek-v4-flash"]
    qwen = CANDIDATES["qwen37-flash"]
    muse = CANDIDATES["muse-glimmer-30b"]

    assert deepseek.model == "deepseek/deepseek-v4-flash-0731"
    assert deepseek.provider_route["only"] == ["coreweave/fp8"]
    assert qwen.model == "qwen/qwen3.7-flash"
    assert qwen.provider_route["only"] == ["alibaba"]
    assert muse.model == "meta/muse-glimmer-30b"
    assert muse.provider_route["only"] == ["deepinfra/bf16"]
    assert deepseek.provider_route["allow_fallbacks"] is False
    assert qwen.provider_route["allow_fallbacks"] is False
    assert muse.provider_route["allow_fallbacks"] is False


def test_candidate_state_is_separate() -> None:
    candidates = list(CANDIDATES.values())

    assert len({item.card_name for item in candidates}) == len(candidates)
    assert len({item.approval_name for item in candidates}) == len(candidates)
    assert len({item.qualification_name for item in candidates}) == len(candidates)
    assert len({item.run_slug for item in candidates}) == len(candidates)


def test_candidate_profile_omits_nonportable_reasoning_effort() -> None:
    for candidate in CANDIDATES.values():
        configure(candidate)
        profile = run_live_ls20._provider_profile()
        assert profile["reasoning_effort"] is None
        assert profile["drop_params"] is True


def test_all_multicandidate_cards_fit_frozen_limit() -> None:
    repo = run_live_ls20.REPO
    cards = [
        repo / "experiments/cards/0158-deepseek-v4-flash-coreweave-preflight-v32.md",
        repo / "experiments/cards/0158-qwen37-flash-preflight-v32.md",
        repo / "experiments/cards/0158-muse-glimmer-30b-preflight-v32.md",
        *(repo / "experiments/cards" / item.card_name for item in CANDIDATES.values()),
    ]

    assert all(Path(card).is_file() and len(Path(card).read_bytes()) <= 1024 for card in cards)
