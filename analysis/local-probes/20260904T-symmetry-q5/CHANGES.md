# q5 — exact change set

Six changes. Companion: `PLAN.md`, `ORACLE-ls20.md`.
Base: `../20260904T-host-control-loop-q3/patched-source/after/solver_agent.py` **plus
whatever q4 applies** — q4 is running; rebase onto its `after/` when it lands.
Line numbers are against the q3c `after/` file; re-anchor on the text, not the number.

One factor: *the comparison capability is present and discoverable.* With
`ARC_SEED_HELPERS` unset every change below is inert and the chassis is q4's.

---

## C1 — seed read-only capability modules (`solver_agent.py`)

### C1a. import site — `with hidden:` block, line 41

The block exists because line 26 says *"Module-level imports are visible to
LLM-generated CodeAct cells"*. `hashlib` must go inside it, never at module level.

```python
with hidden:
    import hashlib          # q5/C1: seed digests; hidden from cells like os/io
    import importlib.util
    import io
    import os  # module-level only (hidden from cells; cells' os access stays blocked)
```

### C1b. call site — `ArcSolverBase.__init__`, line 605

```python
        for d in (self.workspace, self.helpers_dir, self.ipc_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._seed_helpers()                      # q5/C1
        self._states_path.touch()
```

### C1c. new method — place immediately after `__init__`

```python
    @hidden
    def _seed_helpers(self) -> None:
        """q5/C1: copy read-only capability modules into helpers/ before turn 1.

        `ARC_SEED_HELPERS` is an os.pathsep-joined list of source files. Unset, this is a
        no-op and the run is q4's chassis exactly — that is what makes q5 one factor.

        The seed passes the same `_check_helper_ast` guard as a written helper, so the L1
        isolation argument is unchanged: a seeded module is pure computation, no file,
        network, process or dunder access. Verified for symmetry_template.py, which
        imports nothing at all.

        Digests are written to ipc/seeded_helpers.json so the scorer can prove the run
        actually carried the bytes it claims, and can detect a mid-run overwrite (C3
        should make that impossible; record it anyway).
        """
        self._seeded_helpers: dict[str, str] = {}
        spec = os.environ.get("ARC_SEED_HELPERS", "")
        if not spec:
            return
        for path in (Path(p) for p in spec.split(os.pathsep) if p):
            if not _HELPER_NAME.match(path.name):
                raise ValueError(f"seed {path.name!r} must match [a-z0-9_]+.py")
            src = path.read_bytes()
            err = _check_helper_ast(src.decode())
            if err is not None:
                raise ValueError(f"seed {path.name} fails the helper guard: {err}")
            (self.helpers_dir / path.name).write_bytes(src)
            self._seeded_helpers[path.name] = hashlib.sha256(src).hexdigest()
        (self.ipc_dir / "seeded_helpers.json").write_text(
            json.dumps(self._seeded_helpers, indent=2)
        )
```

**Fail loud, not soft.** A missing or guard-failing seed raises at construction. A run
that silently lost its treatment and scored a null would be worse than no run.

---

## C2 — keep the "NONE YET" pressure alive (`_helpers_status_line`, line 1024)

**This is the change that stops q5 being two factors, and it is easy to miss.** The
existing method globs `helpers/*.py`. Seeding `symmetry.py` makes `mods` non-empty from
turn 0, so `helpers: NONE YET — the plan has to live in code…` never fires again. That
line is a q3 signifier that pushed the model to write a world model. Pooling seeded with
authored modules would silently delete it, and q5 would be measuring *symmetry plus the
loss of q3's nudge*.

Replace the body's tail (from `mods = []` through the final `return`):

```python
        authored, provided = [], []
        for p in sorted(self.helpers_dir.glob("*.py")):
            try:
                src = p.read_text()
            except Exception:
                continue
            fns = [
                ln.split("def ", 1)[1].split("(")[0]
                for ln in src.splitlines()
                if ln.startswith("def ")
            ]
            # Public names only, matching load_helpers()'s own filter. Staging the seed
            # showed the unfiltered list advertising _normalise, _to_origin, _label_canon,
            # _validate_colours, _primitive_at_scale and _norm — six private functions with
            # internal signatures, handed to an executor that uses vocabulary literally.
            fns = [f for f in fns if not f.startswith("_")]
            entry = f"{p.stem}.{{{','.join(fns)}}}" if fns else p.stem
            (provided if p.name in self._seeded_helpers else authored).append(entry)

        lines = []
        if provided:
            # Full function list, never truncated. The one law this executor has
            # demonstrated twice is that it uses the vocabulary it is handed literally
            # (canon(point='full') 0/2; rationale "prediction..." 0/24). A truncated
            # list is a list of the wrong names.
            lines.append(
                "provided: " + " | ".join(provided)
                + "  (read-only; self.h.<module>.<fn> after load_helpers())"
            )
        if authored:
            lines.append(
                "helpers: " + " | ".join(authored)
                + "  (self.h.<module>.<fn> after load_helpers())"
            )
        else:
            lines.append(
                "helpers: NONE YET — the plan has to live in code before it can constrain a batch"
            )
        return "\n".join(lines)
```

---

## C3 — the seeded bytes are read-only (`write_helper`, line 1270)

Without this the model can overwrite `symmetry.py` mid-run and the treatment is gone with
no record. Insert directly after the `_HELPER_NAME` check:

```python
        if not _HELPER_NAME.match(filename):
            return f"REJECTED: filename {filename!r} must match [a-z0-9_]+.py"
        if filename in self._seeded_helpers:                       # q5/C3
            self._log_rejection("seeded_helper_overwrite", [], filename)
            return (
                f"REJECTED: helpers/{filename} is provided read-only. "
                "Write your world model to a different module."
            )
```

---

## C4 — one pointer sentence, and **no** docstring rewrite

Add to the `provided:` line in C2, after the function list:

```python
            lines.append(
                "provided: " + " | ".join(provided)
                + "  (read-only; self.h.<module>.<fn> after load_helpers())"
                + "\n  symmetry compares two spatial arrangements: two figures are the same"
                " — under quotients you choose — iff their canonical forms are ==."
                " A figure is a dict {(y, x): colour}."
                " read_helper('symmetry.py') for the four quotients and the two traps."
            )
```

**Why the pointer and not the brief.** The 40-line teaching text is already the module's
own docstring, and in the uptake run the actor read it unprompted — *"Let's inspect
symmetry.py docstring"* — then chose its quotients correctly on the first attempt. Putting
the brief in the system prompt instead would spend tokens every turn and, worse, leave
static text asserting a capability that is absent whenever `ARC_SEED_HELPERS` is unset.
The docstring ships with the bytes, so text and capability cannot come apart.

**Do NOT edit `MdArcSolverAgent.__doc__` or the per-turn workflow list.** Adding a step 3
clause about comparison would be a second signifier and would tell the actor *where to
point* the tool. q5 establishes availability; the goal-form hint is the q6 arm (PLAN §15).

---

## C5 — `launch_q5.sh`

Copy `../20260904T-goal-candidate-q4/launch_q4.sh` and change exactly two things:

```bash
export ARC_SEED_HELPERS="$PROBE/seed/symmetry.py"
```

placed with the other `export`s, and `--tag q5` in place of `--tag q4`.

**Never point this at `symmetry_template.py`.** An earlier draft of this file did, one
paragraph above the correction. `_seed_helpers` keeps `path.name`, so that path lands
`self.h.symmetry_template`, every call in the docstring is wrong, and C1 cannot catch it —
the file exists, the guard passes, only the name is wrong. That is a null run that looks
like a clean one. `score_q5.py` must assert the digest below.

`_seed_helpers` keeps `path.name`, so pointing this at `symmetry_template.py` would land
`self.h.symmetry_template` and every call in the docstring would be wrong. The byte-exact
copy under the correct name is already staged:

    analysis/local-probes/20260904T-symmetry-q5/seed/symmetry.py
    sha256 7e0c1f1cdcd0d594b1bd21dee0fd74c9624a5a66141575d2d9f947fc298b433f  (10579 B, seeded)
    sha256 289fd2dceaa89dd5aa1704b7497072d850ade6622e0fd7a413270bdc1e5c8f21  (10570 B, Tycho original)

Two sentences of the docstring were chassis-specific and false here: it named `wmlib` as
"the observation toolbox" and claimed the module is "imported into every run_python call".
Neither exists in NOOA, and an executor that uses vocabulary literally would go hunting for
an unimportable module. Both were rewritten to name `self.load_helpers()` / `self.h.symmetry`
and "any connected component you have extracted yourself". Zero `wmlib` and zero
`run_python` mentions remain; no geometry was touched; the guard still returns None.

Verified against the live guard under that name: `_check_helper_ast(...) -> None`.
Point `ARC_SEED_HELPERS` at that path, and assert the digest in `score_q5.py`.

Everything else identical to q4: same game, variant, model, skill, sandbox flags,
`--visual off`, `--effort-ladder ''`, `ARC_LLM_EXTRA_BODY`, turn/step caps, daemonized
launch via a `daemonize_q5.py` copied from `daemonize_q4.py`.

---

## C6 — `score_q5.py`, written and frozen before launch

Copy `score_q4.py`; keep P1–P4, P6, P7 unchanged as hold-the-line; add three clauses.
Do not edit `score_q4.py`.

`reach_scorer.py` in this directory, already written and validated.

**The first draft of S1 could not have detected use.** It looked for `type ==
"repl_execute"`, a `code` field and `status == "complete"`. The real events carry `event`,
`success`, `error`, `code_length`, and a `functions_used` list holding only bare resolved
names — on q4, four distinct names across 399 cells, with `grid_array` absent entirely.
Cell source is not in the events at all; it is in
`agent_logs/**/messages/step_*_round_*_assistant.md` as a `[tool_call] name=execute_python
args={"code": ...}` blob. A gate on the old draft reads 0 whether or not the module was
used, and the null would have been misread as "discoverability failed".

`reach_scorer.py` reads reaches from the messages and execution status from the events,
resolving at **step level**: a reach in a step with zero failed cells executed; a reach in
a step containing any failure must be pinned by an exact-unique `code_length`, and an
unresolvable one scores ERR. Ordering cannot be used — the framework prepends prefill
cells that appear in no message, and the per-step surplus on q4 was 2 for 46 steps, 1 for
10 and 3 for 6. Undercounts are possible; inflation is not, which is the property the
original uptake scorer existed to protect.

Validated both directions on q4's trace before launch:

    --module symmetry --probe grid_array
      -> reached=0 executed=0;  probe found in 218/279 cells
    --module self
      -> reached=272 executed=272, 15 distinct functions

Absent module reads 0; heavily-used module reads 272. Re-run both after any rebase.

- **S1** = `len(executed_reaches(...)) >= 1`. Prior 0 — unreachable in every NOOA run.
- **S2** = at least one turn that contains an executed reach **and** whose
  `submit_actions` carried a `predicted_path` of **length ≤ 5** with an `expect` in
  `{level_up, terminal}` that the scorer's own `score_predicted_path` can score. Binary,
  stated as binary, never averaged across runs.

  The length bound is q4's cheap-discriminator bound and it is load-bearing. Without it,
  S2 is "they called symmetry in a turn where they also hoped to win" — q4 alone had an
  8-action `level_up` plan on turn 2. A discriminator is a cheap test of the objective,
  not a long route that happens to end in a terminal expect.
- **S3** (recorded, not gated) = does any written helper's `Z_SCHEMA` gain a field
  derived from a figure comparison. q3c's was `{grid, mover, mover_size, items}`.

Also verify and report, every run: `ipc/seeded_helpers.json` matches the seed digest, and
`rejections.jsonl` contains no `seeded_helper_overwrite`.

**Oracle clause.** `ORACLE-ls20.md` may be used to *describe* what the actor found — did
its comparison land on the key glyph — but no gate threshold may depend on it. If any
does, the card must say so. Scoring with an oracle label is permitted by `CONTEXT.md`;
tuning a gate to a known answer is not.

---

## Not in q5

- `MdArcSolverAgent` docstring and the per-turn workflow (C4).
- The H2 row-vs-cell no-impact defect (PLAN §14.2) — real, one suppressed door event on
  q3c, and the naive repair over-corrects. Its own card.
- The planner port (`plan_bfs` and friends) — next audit item, its own card.
- Any change to q4 while it runs.
