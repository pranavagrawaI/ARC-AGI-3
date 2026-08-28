"""The geometry library, checked against mathematics rather than against recordings.

Game-derived helpers were abandoned because each one turned out to be required by exactly
one game and invisible to the others, so a corpus could never say whether an abstraction
generalised. Theorems can. Every check below can fail, and several did during construction:
without the minimisation over the group `canon` is not invariant and Burnside reports 97
orbits where there are 20.

These run against `symmetry_template.py` — the exact bytes seeded into the workspace — so
what the model receives is what the theorems were proved about.
"""

from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path

import pytest

from tycho_nooa import symmetry_template as S

GROUP = S.FULL


def fig(cells, labels=None):
    return {c: (labels[i] if labels else 1) for i, c in enumerate(cells)}


def blow(f, k):
    """The k-fold magnification: one cell becomes a k x k block."""
    return {(y * k + i, x * k + j): label
            for (y, x), label in f.items() for i in range(k) for j in range(k)}


def eq(a, b, point=S.FULL, **kw):
    """Equivalence stated only as equality of canonical forms, never through a named
    relation — so a test of the quotient cannot be satisfied by the name."""
    return S.canon(a, point, **kw) == S.canon(b, point, **kw)


@pytest.fixture(scope="module")
def figures():
    rng = random.Random(0)
    cells = [(y, x) for y in range(4) for x in range(4)]
    return [{c: rng.randrange(3) + 1 for c in rng.sample(cells, 5)} for _ in range(120)]


# --- is it a group at all ---------------------------------------------------------

def test_d4_satisfies_the_group_axioms():
    assert len(GROUP) == 8
    assert all(S.compose(g, h) in GROUP for g in GROUP for h in GROUP)
    assert all(S.compose(g, S.I) == g == S.compose(S.I, g) for g in GROUP)
    assert all(any(S.compose(g, h) == S.I for h in GROUP) for g in GROUP)
    assert all(S.compose(S.compose(a, b), c) == S.compose(a, S.compose(b, c))
               for a in GROUP for b in GROUP for c in GROUP)


def test_every_element_is_a_lattice_bijection():
    """det = +-1 is exactly the condition of being a bijection of the lattice, which is why
    these eight are all of them and D4 is not a modelling preference."""
    assert all(abs(a * d - b * c) == 1 for (a, b), (c, d) in GROUP)


def test_closure_is_computed_from_arbitrary_generators():
    """The model must be able to name the group ITS game's actions generate, not choose
    from a menu of four. D4 has exactly ten subgroups."""
    subs = {S.generate([g, h]) for g in GROUP for h in GROUP}
    assert all(all(S.compose(x, y) in H for x in H for y in H) for H in subs)
    assert all(len(GROUP) % len(H) == 0 for H in subs), "Lagrange"
    assert len(subs) == 10


def test_it_is_a_group_action(figures):
    assert all(S.act(S.I, f) == f for f in figures)
    assert all(S.act(S.compose(g, h), f) == S.act(g, S.act(h, f))
               for f in figures[:15] for g in GROUP for h in GROUP)


# --- is the canonical form right --------------------------------------------------

@pytest.mark.parametrize("point", [S.FULL, S.ROTATIONS, S.HALFTURN])
def test_canon_is_invariant_under_its_own_group(figures, point):
    assert all(S.canon(S.act(g, f), point) == S.canon(f, point)
               for f in figures for g in point)


def test_canon_is_sound_and_complete(figures):
    """Equal canonical forms iff some group element actually carries one to the other.
    Completeness is what lets normalising both sides replace searching for a transform."""
    for a in figures[:40]:
        for b in figures[:40]:
            assert bool(S.transporter(a, b)) == (S.canon(a) == S.canon(b))


def test_orbit_stabiliser(figures):
    for f in figures:
        for H in (S.FULL, S.ROTATIONS, S.HALFTURN):
            assert len(S.orbit(f, H)) * len(S.stabiliser(f, H)) == len(H)


def test_the_transporter_is_empty_or_a_coset(figures):
    """Which is why "the" transform carrying a onto b is often not a well-formed idea: a
    figure with a symmetry of its own is carried there by more than one element."""
    for a in figures[:40]:
        for b in figures[:40]:
            assert len(S.transporter(a, b)) in (0, len(S.stabiliser(a)))


def test_burnside():
    """Orbit count equals the average number of figures each element fixes. This is the
    check that fails loudest when `canon` stops canonicalising."""
    box = [(y, x) for y in range(3) for x in range(3)]
    univ = {tuple(S.canon(fig(c), S.NONE)) for c in combinations(box, 4)}
    figs = [{(y, x): label for y, x, label in u} for u in univ]
    orbits = {S.canon(f) for f in figs}
    fixed = sum(1 for g in GROUP for u, f in zip(univ, figs)
                if tuple(S.canon(S.act(g, f), S.NONE)) == u)
    assert (len(univ), len(orbits)) == (97, 20)
    assert len(orbits) * len(GROUP) == fixed


# --- the four quotients, each one load-bearing -------------------------------------

def test_the_point_quotient_refines_correctly(figures):
    for a in figures:
        for b in figures:
            if eq(a, b, S.NONE):
                assert eq(a, b, S.ROTATIONS)
            if eq(a, b, S.ROTATIONS):
                assert eq(a, b, S.FULL)
            if eq(a, b, S.FULL):
                assert eq(a, b, S.FULL, scale=True)
                assert len(a) == len(b)
                assert sorted(a.values()) == sorted(b.values())


def test_the_translation_quotient_can_be_declined(figures):
    """Comparing two OBJECTS and comparing two SCENES are different geometries, and the
    library must be able to say both: normalise each half of a board to its own bounding
    box and "same shape, wrong half" reads as a match."""
    f = figures[0]
    shifted = {(y + 3, x + 1): label for (y, x), label in f.items()}
    assert eq(f, shifted)
    assert not eq(f, shifted, translate=False)
    assert all(S.canon(S.act(g, h), translate=False) == S.canon(h, translate=False)
               for h in figures for g in GROUP), "still invariant about the origin"


def test_the_scale_quotient(figures):
    """Magnification is not invertible on a lattice — there is no dilation group to
    quotient by, only the normal form `primitive`."""
    for f in figures[:40]:
        assert eq(f, blow(f, 2), scale=True) and not eq(f, blow(f, 2))
        assert eq(blow(f, 2), blow(f, 3), scale=True)
        core, k = S.primitive(blow(f, 3))
        assert k % 3 == 0 and S.primitive(core)[1] == 1


def test_the_colour_quotient():
    a = fig([(0, 0), (0, 1), (1, 0)], [1, 1, 2])
    b = fig([(0, 0), (0, 1), (1, 0)], [2, 2, 1])
    assert S.canon(a) != S.canon(b)
    assert S.canon(a, colours="any") == S.canon(b, colours="any")


# --- chirality, computed two independent ways --------------------------------------

def test_chirality_agrees_with_the_stabiliser(figures):
    improper = S.FULL - S.proper(S.FULL)
    for f in figures:
        assert S.chiral(f) == (not S.stabiliser(f, S.FULL) & improper)


def test_the_tetromino_anchors():
    s = fig([(0, 1), (0, 2), (1, 0), (1, 1)])
    z = fig([(0, 0), (0, 1), (1, 1), (1, 2)])
    t = fig([(0, 0), (0, 1), (0, 2), (1, 1)])
    o = fig([(0, 0), (0, 1), (1, 0), (1, 1)])
    assert S.chiral(s) and not S.chiral(t) and not S.chiral(o)
    assert S.enantiomorphic(s, z)
    assert S.congruent(s, z) and not S.directly_congruent(s, z)
    assert len(S.stabiliser(o)) == 8 and len(S.stabiliser(s)) == 2


# --- the trap the docstring warns about, as a theorem ------------------------------

def test_congruent_parts_do_not_make_a_congruent_whole():
    """Canonicalising each colour class on its own discards the relative position of the
    classes, so it is strictly coarser than canonicalising the labelled figure. Three games
    were needed to find this empirically; it is one line of group theory."""
    a = fig([(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 2, 2])
    b = fig([(0, 0), (0, 1), (1, 0), (1, 1)], [2, 2, 1, 1])
    parts = all(S.canon({c: 1 for c, label in a.items() if label == k})
                == S.canon({c: 1 for c, label in b.items() if label == k}) for k in (1, 2))
    assert parts
    assert S.canon(a, S.NONE) != S.canon(b, S.NONE)


# --- the wiring: a helper the model cannot find is a helper that does not exist -----

def test_the_agent_seeds_symmetry_beside_wmlib(tmp_path):
    """Seeded by us, not by upstream: `GameWorkspace._seed_worldmodel` rewrites wmlib.py
    from its own template on every fresh start, so a merged file would be overwritten."""
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    seeded = agent.workspace_dir / "symmetry.py"
    assert (agent.workspace_dir / "wmlib.py").exists()
    assert seeded.read_text() == Path(S.__file__).read_text()
    assert (agent.workspace_dir / ".trusted-runtime.json").exists()


@pytest.mark.parametrize(
    "path",
    [
        "dynamics.py",
        "wmlib.py",
        "symmetry.py",
        ".trusted-runtime.json",
        "notes/../symmetry.py",
    ],
)
def test_models_cannot_modify_the_trusted_runtime(tmp_path, path):
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    with pytest.raises(PermissionError, match="trusted runtime"):
        agent._tool_surface.write_file(path, "model-authored")


def test_run_python_mounts_the_trusted_runtime_read_only(tmp_path):
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent
    from tycho_nooa.trusted_runtime import inspect_runtime

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    declared = inspect_runtime(agent.workspace_dir).runtime

    result = agent._tool_surface.run_python(
        "open('symmetry.py', 'w').write('# model-authored\\n')"
    )

    assert "PermissionError" in result or "Read-only file system" in result
    assert inspect_runtime(agent.workspace_dir).runtime == declared


def test_run_python_cannot_downgrade_the_manifest_then_edit_a_helper(tmp_path):
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent
    from tycho_nooa.trusted_runtime import inspect_runtime

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    declared = inspect_runtime(agent.workspace_dir).runtime

    deletion = agent._tool_surface.run_python(
        "import os; os.unlink('.trusted-runtime.json')"
    )
    mutation = agent._tool_surface.run_python(
        "open('symmetry.py', 'w').write('# model-authored\\n')"
    )

    assert (
        "PermissionError" in deletion
        or "Device or resource busy" in deletion
        or "Read-only file system" in deletion
    )
    assert "PermissionError" in mutation or "Read-only file system" in mutation
    assert inspect_runtime(agent.workspace_dir).runtime == declared


def test_legacy_checkpoint_notice_does_not_claim_symmetry(tmp_path):
    from tycho_nooa import runtime_notice

    (tmp_path / "symmetry.py").write_text("# historical and unverified\n")

    assert "legacy/unknown" in runtime_notice.prompt_brief(tmp_path)
    tool_note = runtime_notice.tool_note(tmp_path)["run_python"]
    assert "unverified" in tool_note
    assert "None" in tool_note


def test_resumed_legacy_workspace_reports_unknown_runtime(tmp_path):
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent

    first = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="legacy-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    (first.workspace_dir / ".trusted-runtime.json").unlink()

    resumed = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="legacy-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        resume_workspace=True,
    )

    assert "legacy/unknown" in resumed._resolve_system_prompt()
    run_python = next(
        tool for tool in resumed._tool_surface.native_tools() if tool.name == "run_python"
    )
    assert "unverified" in run_python.description


def test_run_python_preloads_symmetry(tmp_path):
    """Upstream preloads `np`, `wmlib` and `grid`; a module the model must remember to
    import is a module it will not use, so `symmetry` joins them."""
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    out = agent._tool_surface.run_python(
        "print(symmetry.congruent({(0,0):1,(0,1):1}, {(0,0):1,(1,0):1}))"
    )
    assert "True" in out, out


def test_run_python_survives_a_workspace_without_symmetry(tmp_path):
    """The guard exists so a bare or restored workspace cannot break every run_python."""
    from tycho_nooa.wm_probe import GatedToolExecutor

    assert "except Exception" in GatedToolExecutor._PREAMBLE
    assert "import symmetry" in GatedToolExecutor._PREAMBLE


def test_run_python_tells_the_model_symmetry_is_there(tmp_path):
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    described = {t.name: t.description for t in agent._tool_surface.native_tools()}
    assert "symmetry.canon" in described["run_python"]
    assert "wmlib" in described["run_python"], "upstream's text must still be there"


def _agent_and_builder(tmp_path):
    from nooa.unifiedllm import FakeLLMClient

    from tycho_nooa import TychoNooaAgent
    from tycho_nooa.builder import FocusedBuilder

    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    builder = FocusedBuilder(
        llm=FakeLLMClient(scripted_responses=[]),
        executor=agent._tool_executor,
        trace_path=None,
    )
    return agent, builder


def test_both_system_prompts_name_symmetry(tmp_path):
    """The tool description is read when choosing a tool; the system prompt is read when
    forming a hypothesis, which is where this library is either reached for or not."""
    agent, builder = _agent_and_builder(tmp_path)
    for prompt in (agent._resolve_system_prompt(), builder._resolve_system_prompt()):
        assert "symmetry.py" in prompt
        assert "symmetry.canon" in prompt


def test_the_builder_prompt_is_appended_to_not_replaced(tmp_path):
    """Upstream's builder.system enumerates run_python's preloads as a CLOSED list written
    before symmetry existed. We append a correction; clobbering upstream would be worse than
    the gap it fixes."""
    _, builder = _agent_and_builder(tmp_path)
    prompt = builder._resolve_system_prompt()
    assert "wmlib.death_events()" in prompt, "upstream's own handles must survive"
    assert "any preload list you were given elsewhere" in prompt


def test_the_builder_shares_the_actors_symmetry_aware_tool_surface(tmp_path):
    """One executor serves both, so a preload that reached only the actor would be a silent
    asymmetry between the two readers of the same workspace."""
    agent, builder = _agent_and_builder(tmp_path)
    for surface in (agent._tool_surface, builder._tool_surface):
        run_python = next(
            t for t in surface.native_tools() if t.name == "run_python"
        )
        assert "symmetry.canon" in run_python.description


def test_the_three_channels_have_one_source(tmp_path):
    """Three places tell the model about symmetry. They drift unless they share a source."""
    from tycho_nooa import runtime_notice

    agent, builder = _agent_and_builder(tmp_path)
    described = {t.name: t.description for t in agent._tool_surface.native_tools()}
    assert described["run_python"].endswith(runtime_notice.TOOL_NOTE["run_python"])
    assert agent._resolve_system_prompt().endswith(runtime_notice.PROMPT_BRIEF)
    assert builder._resolve_system_prompt().endswith(runtime_notice.PROMPT_BRIEF)


# --- what the first live run broke on ------------------------------------------------
# Run tycho-nooa-symmetry-uptake-ar25-20260820T101248Z reached for `symmetry` four times.
# The two calls using the presets worked; both calls passing `point` as a string raised
# `ValueError: not enough values to unpack` from inside `act`. These replay them.

@pytest.mark.parametrize("name, expected", [
    ("full", S.FULL), ("rotations", S.ROTATIONS),
    ("halfturn", S.HALFTURN), ("none", S.NONE),
    ("FULL", S.FULL),  # the docstring names the constants in caps; accept that spelling too.
])
def test_a_point_group_may_be_named(name, expected):
    fig = {(0, 0): 1, (0, 1): 1, (1, 0): 2}
    assert S.group(name) == expected
    assert S.canon(fig, point=name) == S.canon(fig, point=expected)

def test_the_live_failures_now_answer():
    """Builder calls 25 and 41 of the run, reduced to their essentials."""
    a = {(0, 0): 4, (1, 0): 4, (1, 1): 4}
    b = {(5, 5): 4, (6, 5): 4, (6, 6): 4}
    kw = {"point": "full", "translate": True, "scale": False, "colours": "fixed"}
    assert S.canon(a, **kw) == S.canon(b, **kw)

def test_naming_a_group_that_is_not_one_says_which_names_exist():
    """`point='free'` was the run's other mistake: position freedom read as a pose group.
    An error that does not name the alternatives just costs another turn."""
    with pytest.raises(ValueError) as e:
        S.canon({(0, 0): 1}, point="free")
    msg = str(e.value)
    for name in ("'full'", "'rotations'", "'halfturn'", "'none'", "generate("):
        assert name in msg
    assert "translate" in msg, "the confusion to correct is pose vs position"

def test_every_entry_point_taking_a_group_accepts_a_name():
    """A coercion that reaches only `canon` would leave the same trap one call deeper."""
    fig = {(0, 0): 1, (0, 1): 1, (1, 0): 1}
    assert S.stabiliser(fig, point="full") == S.stabiliser(fig, point=S.FULL)
    assert S.transporter(fig, fig, point="rotations") == S.transporter(fig, fig, S.ROTATIONS)
    assert S.orbit(fig, point="halfturn") == S.orbit(fig, point=S.HALFTURN)
    assert S.proper("full") == S.proper(S.FULL)

def test_a_misspelt_colours_raises_rather_than_permuting_in_silence():
    """`colours` had the same defect as `point` but quieter: anything not "fixed" meant
    "any", so a typo silently answered a different question. In a goal test that is worse
    than a traceback."""
    with pytest.raises(ValueError, match="'fixed' or 'any'"):
        S.canon({(0, 0): 1}, colours="fixd")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"point": "free"}, "not a point group"),
        ({"colours": "fixd"}, "'fixed' or 'any'"),
    ],
)
def test_empty_figures_do_not_bypass_option_validation(kwargs, match):
    """Two empty extraction failures compare equal, so validation matters even here."""
    with pytest.raises(ValueError, match=match):
        S.canon({}, **kwargs)


def test_colour_quotient_scales_with_cells_not_label_permutations(monkeypatch):
    """Canonical relabelling is first-occurrence normalisation, not a k! search."""

    def factorial_search_was_used(*args, **kwargs):
        raise AssertionError("colour canonicalisation enumerated label permutations")

    monkeypatch.setattr(S, "permutations", factorial_search_was_used, raising=False)
    fig = {(0, index): label for index, label in enumerate(range(20, 4, -1))}
    expected = tuple((0, index, index) for index in range(16))
    assert S.canon(fig, point="none", translate=False, colours="any") == expected
