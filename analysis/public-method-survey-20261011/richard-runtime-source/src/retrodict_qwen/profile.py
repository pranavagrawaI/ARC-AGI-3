"""Baseline R settings (Retrodict on Qwen3.8); the rest of Config keeps the shared defaults."""

R_DEFAULTS = {
    # Fresh-session threshold: upstream 150K input tokens; here the shared 96K compaction
    # decision (three games share the KV pool). v5.2: the game subtracts output_tokens_cap
    # (one reply of headroom), so a session restarts once its context passes 49,152 tokens.
    "compaction_tokens": 65536,
    # The v4 A/B/C mechanisms (object view, transition summary, dump suppression, no-change
    # repeat rejection, per-response action cap) are not part of this faithful port.
    "object_view": False,
    "transition_summary": False,
    "suppress_grid_dumps": False,
    "reject_no_change_repeats": False,
    "actions_per_turn": None,
    # Upstream python tool: 30 s cap; the executor clips by the remaining clock.
    "bash_seconds": 30,
    "bash_max_seconds": 30,
    # Upstream's analysis interpreter had numpy (plus scipy/networkx, not staged here).
    "scientific_python": True,
    # v5 (18 September): upstream allowed 128 requests per invocation and no time budget; one
    # v3 invocation spent 429 s and 14 requests on a seven-click plan. The eighth request is
    # the invocation's last (tool-less), or the first after the time budget (capped at a
    # fraction of the remaining clock late in a game); replies are capped per request.
    # v5 ran 150 s / 40% / 8,192 tokens: the cap fired four times, shrank to 32 s late in
    # games and cut four xhigh replies (22-25K output tokens per invocation), so v5.1
    # (user decision, 18 September) loosens it to 240 s / 60% / 16,384 with 8 requests.
    "call_turns": 8,
    "invocation_budget_seconds": 240,
    "output_tokens_cap": 16384,
    # Upstream action_cap (the pilot protocol); the 600 s clock ends games first.
    "max_actions": 2000,
}
