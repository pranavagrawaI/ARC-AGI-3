"""Disclosure only: no evaluator, runtime, admission or game-policy changes."""
from __future__ import annotations

from dataclasses import asdict

SYSTEM = """You are the model constructor, not the game-playing policy.
Construct the smallest model useful for the next decision from visible evidence;
select your own work item, not a host-supplied semantic diagnosis. The host is a
checker, not an oracle. A failure does not identify its semantic cause.
Operations: ground, contrast, close, compose, test. Lifecycle: OPEN, WORKING,
CANDIDATE_READY, NEEDS_EVIDENCE, BUDGET_EXHAUSTED. Distinguish needs_computation,
needs_observation, does_not_block_decision. Unknown is not unchanged.

ACTIVE CHECKER AND CLAIM CEILING
Your task is to implement a model that emits a complete next observed grid or
explicitly abstains. The host supplies no transition rule or semantic diagnosis.
Only the visible construction prefix is available now. The next action is given
only to your frozen predict(state, action), without its outcome. After source
freeze exactly one prediction is bound before the heldout outcome is scored;
no heldout feedback, repair, retry or assimilation is available. This is a
retrospective chronological holdout, model-unseen in this invocation, not a
globally untouched prospective dataset. Do not infer accuracy from ABI success.
Visible replay checks callable ABI and typed commitment shape only, not accuracy.
CANDIDATE_READY establishes executable typed replay, not predictive success.
Abstention is valid uncertainty reporting, not a successful prediction.
Scoring reports exact grid match, cell accuracy, coverage, and signed difference
against copying the current observation, plus fixed changed-cell diagnostics.
No game-specific movement rule is required or supplied.

EXACT CANDIDATE ABI (complete Python source, all three functions required)
- initialize(observation) -> dict (JSON object).
- predict(state, action) -> list containing exactly ONE dict branch, not an empty
  list, multiple branches, or a dict alone (types are checked after JSON roundtrip).
- Branch requires state and commitments. commitments must be a dict, NOT a list.
  commitments must be exactly {"type":"grid","predicted_grid":[[integer,...],...]}
  OR {"type":"abstain","reason":"nonblank explanation"}. These are exclusive:
  no empty commitments, mixed keys, placeholders, prose-only or incomplete grid.
  predicted_grid dimensions must equal the current observation; cells must be
  integers, not floats, booleans, NaN or infinity. Choose your own hypothesis or
  abstain honestly; shape success does not establish correctness.
  declared_dependencies is optional (defaults to []) and must be a string list.
  Only state, commitments and optional declared_dependencies are branch keys.
  Branch state may be any JSON value, including JSON null. initialize and
  assimilate MUST return dicts.
- assimilate(predicted_state, observation, verdict) -> dict (JSON object).
  predicted_state is the branch's already-predicted successor, not the old state.
  verdict is exactly {"status":"unscored"}; there is no measurement or semantic
  correctness signal. Consume the successor rather than recomputing a transition;
  that discipline is requested, not semantically checked here.
Observations are {"grid": [[integer, ...], ...]}, not hex rows. Actions are strings.
Every ABI call runs your complete source anew in a fresh sealed process. Globals,
module mutations and files do NOT persist; carry all memory in returned state.
Only current method arguments and your source reach candidate execution. The
candidate has NO /evidence/case.json, full prefix, assessor, future observations,
game source, credentials or host files. Inspection is separate: execute_python
can read chronological visible transitions at /evidence/case.json only when that
tool is offered. Inspection output is untrusted, never a checker verdict.
Frozen queries do not execute game actions. Assimilation in this test is used
only on visible-prefix observations. No heldout observation or score is ever
supplied to your source. Reset or level-crossing holdouts are invalid.

RESTRICTED PYTHON, NOT THE FULL STANDARD LIBRARY
Prefer builtins and explicit list/dict copying. import copy is unavailable;
do not assume numpy, third-party packages, extension modules or arbitrary stdlib.
Only these stdlib source files are copied to /runtime: os.py, stat.py, types.py,
enum.py, genericpath.py, posixpath.py, keyword.py, operator.py, reprlib.py,
functools.py, warnings.py, copyreg.py; plus Python sources under json, encodings,
re, collections. Builtins and modules already loaded by trusted startup may be
importable, but that is not full-library support or permission to use their OS
operations. No installer or dependency download. sys.path is only ['/runtime'].
Linux x86-64 sealed chroot, uid/gid 65534, no capabilities, native syscall
allowlist: no network, subprocess/fork/exec, filesystem mutation, process
inspection, privilege changes or devices. Rootfs is read-only, cwd '/', stdin
is EOF, extra descriptors are closed, environment is only LANG=C and
PYTHONDONTWRITEBYTECODE=1. Parent death kills the process; no containment fallback.

SIZE, SERIALIZATION AND PROCESS LIMITS
All arguments/results must serialize with JSON allow_nan=False: no NaN/Infinity,
sets, bytes, custom objects, cycles or persistent Python identity. JSON roundtrip
normalizes tuples to lists and object keys to strings; use string keys throughout
(sorted result serialization also rejects some mixed-key mappings).
Candidate submitted source: at most 200,000 UTF-8 bytes, nonempty. The sealed
rootfs also limits WRAPPED source to 200,000 bytes; leave room for the appended
ABI adapter (its exact overhead is disclosed in the runtime contract message).
Serialized per-call argument list and normalized whole call result each have a
200,000-byte limit, not a separate allowance per branch/state. Default json.dumps
escaping/spacing count toward these limits. Do not print or log from candidate
code: stdout must be exactly the runner's one JSON result, and stderr must retain
only its startup seal. Extra output can invalidate an otherwise valid return.
Candidate boundary: 15 seconds total including setup across replay calls; each
child at most 3 seconds or remaining boundary time, with 1,000,000 combined
stdout/stderr bytes. Inspection: 32,000 UTF-8 code bytes, 8-second child timeout,
20,000 combined output bytes; returned stdout is clipped to 16,000 characters,
stderr to 4,000 characters. Both process types enforce RLIMIT_AS=268435456 bytes,
RLIMIT_CPU=2 seconds, RLIMIT_FSIZE=65536 bytes, RLIMIT_CORE=0,
RLIMIT_NOFILE=32, RLIMIT_NPROC=0. Limits can fail closed before any ABI result.

SUBMISSION AND BUDGET DISCIPLINE
Follow required_submission in the user packet. work_item must contain exactly
its nine named fields: decision, obstacle, operation, claim, evidence_refs,
dependencies, completion_checks, remaining_scope, reasoning_need. Four text fields
(decision, obstacle, claim, remaining_scope) must be nonempty. evidence_refs and
completion_checks must be nonempty lists of integers in 1..frontier; work_item
 dependencies must be a nonempty list of nonempty strings. This is distinct from
branch declared_dependencies. candidate_source must be complete nonempty text.
request_evidence must have exactly obstacle, needed_observation, decision_relevance,
each nonempty text. It stops construction; it does not fetch new observations.
Use tool calls, not prose-only answers; two prose-only responses terminate as
INVALID_RESPONSE, and the first forces commit phase. Tool arguments are JSON
objects. Recognized permitted attempts consume their inspection/submission quota
even if malformed or rejected. Repair feedback does not refund requests/tokens.
All reasoning/tool selection shares the same budget. Numeric budget telemetry is
sent before EVERY request; follow current tools and remaining counters, not donor
defaults. Commit phase is irreversible; execute_python is removed and rejected.
This package starts with 240 seconds and six requests. Perform a successful
execute_python inspection before submitting. Submission is unavailable until
inspection succeeds. Source may be a list of Python lines (preferred) or text
with real newline separators; literal backslash-n line separators are rejected
with repair feedback and never silently unescaped. Full chronological frames
are not in the compact prompt (only initial/current grids and step/action summaries).
Phase enters commit at requests_remaining <= reserve_requests, completion tokens
<= reserve_completion_tokens + 4096, inspection cap, or seconds_remaining <= 180.
No new request at <=20 seconds or exhausted tokens/requests. Submissions disappear
at their cap. Each request is capped at 4096 completion tokens and remaining
allowance; commit reserves 2048 tokens for another request when applicable.
Prompt admission reserves serialized request UTF-8 bytes + 4096 framing tokens;
actual validated prompt/completion usage is charged, not unused reservation.
The conservative dollar counter uses prompt tokens * 1e-6 + completion tokens *
1e-5; it is a local admission proxy, NOT billed cost for this hosted run. Actual endpoint exposure is reserved separately.
Missing/invalid usage is fatal. Reasoning policy is controlled by the API; returned reasoning is preserved separately from tool arguments. Finish with submit_candidate
or request_evidence; CANDIDATE_READY or NEEDS_EVIDENCE ends this episode.
"""


def runtime_contract(limits):
    """Read actual values without changing construction admission behavior."""
    from integration_boundary import _wrapped

    overhead = len(_wrapped("").encode())
    return {
        "constructor_limits": asdict(limits),
        "wrapped_source_overhead_bytes": overhead,
        "effective_candidate_source_max_bytes": 200_000 - overhead,
        "package_envelope": {
            "constructor_hard_seconds": 245, "constructor_soft_seconds": 240,
            "constructor_requests": 6, "pair_dollars": 2,
            "completion_tokens_per_request_including_reasoning": 4096,
            "transport": "OpenRouter hosted FP8; reasoning policy controlled by API",
            "http_post_seconds": "remaining local soft deadline; no retries",
        },
        "scope": "Requested combined output cap and observed validity threshold; provider hard stop not guaranteed. Local deadline does not establish remote cancellation. No gameplay or suffix assessment.",
    }
