# Review candidate — constructor reasoning pair

No paid calls. This directory is a complete runnable candidate, awaiting independent exact-byte review, Hermes release and explicit credential-use approval. It is separate from R10a/R10b. Read `EXPERIMENT-CARD.md`, `DISCOVERY.md`, `AUTH-ACCESS-NOTE.md`, `engine.diff` and the test log.

## Source and runtime

`SOURCE-BINDINGS.json` hashes the read-only R10 donor files. `engine/` contains copies; only constructor protocol/continuation and shared transport disclosures were adapted. `prepare.py` records how copies and prefix were obtained; **do not rerun it on reviewed bytes**. The new `protocol.py`, `transport.py`, `auth.py` and `run_pair.py` implement the isolated hosted adapter, admission ledger and gated once-only launcher. `ORIGINAL-FIRST-MESSAGES.json` is the preserved original start (no historic assistant answers). `PREFIX-BINDING.json` identifies exact actual prefix and original request; the model only receives the copied first prefix.

The inherited engine's transcript records its logical request marker (reasoning off/provider empty), whereas the authoritative outgoing OpenRouter request body is always `raw/NNN.request.json`. `transport.wire()` alone applies arm policy and routing. Do not mistake logical engine markers for treatment evidence. Returned raw reasoning is retained before any projection. The off/on initial shared hashes omit only the declared reasoning object, and are checked again before the production on dispatch. Original system/schema/source formats and evaluator behavior are shared.

Use the existing Python at:
`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/kaggle/.venv/bin/python`

The mapped-user `unshare` boundary is mandatory; model tools use copied original chroot/seccomp runners with empty secret-free child environment. Constructor temporary rootfs files are placed beneath each new arm's `tmp/`. Inspection/code cannot access the host, network, credentials or evaluator. The parent hard-kills the owned worker group on timeout; sealed children have parent-death kill. Only successful timely child exits can promote root freeze/result authority. No provider cancellation is implied.

## Local verification (NO paid request or credential loader)

Run from repository root, with a NEW basetemp for each invocation:

```bash
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/openrouter-constructor-reasoning-20260916T185550Z/test_pair.py --basetemp=analysis/openrouter-constructor-reasoning-20260916T185550Z/verification/reviewer
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/openrouter-constructor-reasoning-20260916T185550Z/check_candidate.py
```

`test_pair.py` uses loopback HTTP fixtures, not a model. The fixture candidate is deliberately an empty-commitment ABI example, not a solution; it is never sent in real arm prompts. Tests retain provider-shaped reasoning details/tool metadata, malformed/source failure feedback, and actual sealed execution. Temporary test results/fictional fixture receipts cannot establish hosted treatment or efficacy.

## External gates (no self-authored approvals)

Reviewer JSON must contain task `OR-CONSTRUCTOR-REASONING-PAIR`, role `independent-reviewer`, decision `PASS`, endpoint `reka/fp8`, and exact `inventory_sha256` from `INVENTORY.json` bytes.
Hermes JSON must contain the same task/endpoint/hash, role `Hermes`, decision `RELEASE`.
A separate PM credential-use JSON must contain the same task, role `PM`, decision `APPROVED`, credential_use `sole-key-existing-loader`. It is not currently present. Until it exists, no credential file is opened by the runnable entry.

Once those independently authored artifacts exist, the exact production command is:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/kaggle/.venv/bin/python -B /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/openrouter-constructor-reasoning-20260916T185550Z/run_pair.py --execute --review /ABSOLUTE/INDEPENDENT-REVIEW.json --release /ABSOLUTE/HERMES-RELEASE.json --credential-approval /ABSOLUTE/PM-CREDENTIAL-APPROVAL.json
```

The three absolute approval paths must be supplied by their owners; the placeholders are intentionally non-executable. No warmup or paid preflight. Do not invoke `--worker` manually. `live/` creation is exclusive; any partial/consumed state means HOLD, never delete/reset/retry. No endpoint change after start. Missing usage/timeout retains unknown exposure and prevents on. Do not edit candidate/card after the first paid call; subsequent diagnosis belongs in separate artifacts.

## Authentication disclosure

`discovery/auth-presence.json` found no exported worker key. Before PM's follow-up arrived, a sole-key loader availability check opened the existing credential file once; only booleans/mechanism were emitted (`discovery/auth-check.json`). No secret was printed/copied or used remotely. That historical record is preserved. PM subsequently clarified credential-file access lacked approval; `auth.py` is now inert by default, and the production entry requires a separate explicit approval before enabling it. This is an authorization gate, not a model/budget feasibility failure.
