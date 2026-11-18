# ARC one-tool continuous-batching runtime

One local reasoning model. One exposed tool, `do_actions`. Independent game
conversations share one vLLM server. No planner, Python execution tool, world
model, critic, retrieval tool, speculative branches, or second model.

## Validation status

The Python files compile and 21 CPU tests pass. They cover concurrent bounded
request admission, independent game data, action-sequence execution, level and
terminal boundaries, malformed/truncated output rejection, deadlines, and no
retry of uncertain action outcomes. The fake backend tests are NOT GPU inference
benchmarks. This environment has neither vLLM nor the ARC SDK installed and has
no usable CUDA GPU. No real model, GPU throughput, native parser round trip, or
Kaggle submission was tested here. The launcher's two-call preflight is intended
to test the actual checkpoint/parser pair on the target machine before games run.

The adapter targets the official upstream Agent/Swarm interface inspected on
2026-09-05. Check the competition-mounted copy against that interface. It uses
the environment Swarm has already created and leaves official discovery,
scorecard creation/closing, endpoint configuration, and game accounting intact.
It does not start second copies of games or request live scores.

## Files

- `one_tool_runtime.py`: protocol, validation, native reasoning response handling,
  a FIFO in-flight request gate, persistent HTTP clients, the sequential game loop,
  and append-only per-game evidence.
- `one_tool_agent.py`: small adapter for the official Swarm; each game thread runs
  the plain tool loop with its own environment. No CUDA objects in game threads.
- `launch.py`: register the adapter in a writable official-runner copy, start one
  local vLLM server, run a two-request tool preflight, start the official runner,
  enforce one shared deadline, and save configuration/server metrics.
- `benchmark.py`: replay development model requests at a specified concurrency.
  It never takes environment actions. This is a serving test, not an ARC score test.
- `test_runtime.py`: CPU tests. Run `python -m pytest -q`.

## Kaggle setup

In the FIRST notebook cell, before installation, copying files or model loading:

```python
import os, time
os.environ.setdefault("ARC_RUN_STARTED_MONOTONIC", str(time.monotonic()))
```

The competition documents a nine-hour notebook runtime limit, not nine hours of
extra inference after setup. The default shared play deadline is the first-cell
timestamp plus nine hours minus a five-minute shutdown reserve. Waiting, loading,
warmup, preflight and inference all consume the same clock. Re-running a cell must
not reset it. This timestamp is valid across processes on the same Linux host.

Use the competition's standard gateway/bootstrap procedure and its provided ARC
packages. Copy the official `ARC-AGI-3-Agents` directory into a writable location,
typically `/kaggle/working/ARC-AGI-3-Agents`. Preserve the competition's forced
operation mode and environment variables. This bundle does not replace gateway
startup or manufacture a submission file.

Attach the model weights/tokenizer and a tested offline wheelhouse as notebook
inputs. Install a vLLM build compatible with that model, GPU, Torch and CUDA, plus
`httpx`. No universal vLLM binary version is asserted here: the checkpoint and
hardware were not specified and were not tested. Freeze the exact successful
wheelhouse after a target-machine smoke test; do not install an untested latest
version in the scored run. `launch.py` records the actual installed versions.

Unzip this directory in `/kaggle/working`. Set the local model path and parser
configuration. The DEFAULT parser preset below is for an original Qwen3 checkpoint
using Hermes-style tool calls. It is not a claim about every newer Qwen release.
Use the checkpoint's documented reasoning parser, tool parser and chat template.

```python
import os, subprocess, sys

os.environ.update({
    "LLM_CONCURRENCY": "16",          # maximum requests admitted at once
    "MAX_NUM_SEQS": "16",             # engine's maximum running sequences
    "MAX_BATCHED_TOKENS": "16384",     # per-engine-iteration token budget
    "MAX_MODEL_LEN": "32768",         # prompt + output context limit
    "MAX_NEW_TOKENS": "16384",        # output safety cap, including reasoning
    "GPU_MEMORY_UTILIZATION": "0.90",
    "REASONING_PARSER": "qwen3",       # checkpoint-specific
    "TOOL_CALL_PARSER": "hermes",      # checkpoint-specific
    "CHAT_TEMPLATE_KWARGS": '{"enable_thinking":true}',
})

subprocess.run([
    sys.executable,
    "/kaggle/working/arc_one_tool_batch/launch.py",
    "--model", "/kaggle/input/YOUR_MODEL/CHECKPOINT_DIRECTORY",
    "--agents-dir", "/kaggle/working/ARC-AGI-3-Agents",
], check=True)
```

These are starting values, NOT a demonstrated optimum. The launcher assumes the
model plus execution workspace plus useful cache capacity fit on one GPU. It uses
one model replica with tensor parallelism 1. Quantization is read from the mounted
checkpoint; it does not quantize a full-precision checkpoint at startup. Additional
engine flags can be supplied as a JSON string array in `EXTRA_VLLM_ARGS_JSON`.

For a development game filter, append `--game ls20` to the launcher command. Do
not filter the final submission's game list. The launcher imports the adapter
under registry name `onetool`; the official command is `python main.py --agent
onetool`. Port 8000 is for the local model, distinct from the usual game gateway
port 8001. There are no external model API requests.

The preflight spends two bounded model calls on fake observations, not scored
ARC environment actions. It checks that the model emits exactly one valid tool
call, accepts the tool result with its native chat template, and emits another.
Inspect `server.log`, the preflight journal, and `agent.log` on failure. For a
previously verified frozen stack, `--skip-preflight` removes these two checks.

## One tool, precise semantics

Example model tool arguments:

```json
{"actions":[{"name":"ACTION1","repeat":3},{"name":"ACTION6","x":12,"y":7}]}
```

The host validates the whole argument list before executing anything, then
executes the expanded primitive actions one by one through the official SDK.
Each action has its own real observation. The model is not called between those
actions. All returned frames, including animation frames, are included in the
single tool result as lossless hexadecimal rows.

Execution stops at the first level change, GAME_OVER, WIN, RESET, unavailable
action, action cap or deadline. Remaining actions are discarded and the result
states how many were executed. GAME_OVER does not automatically spend a RESET;
the model can request RESET through the same tool. A sequence is open-loop within
those boundaries: there is no world model and no claimed prediction-mismatch guard.

`ACTION1` is not assumed to mean UP. Meanings must be inferred. Coordinates are
validated for ACTION6. GameAction enum objects are never mutated with `set_data`:
request-local coordinate dictionaries avoid shared enum state across threads.

## What the runtime does not silently change

- `n=1`; exactly one generation per game decision, not multiple sampled answers.
- Thinking remains enabled for the example parser preset. No separate thinking
  token budget is imposed. The configured output/context/global limits still
  apply. Literal unlimited reasoning cannot fit inside a finite run.
- Full conversation messages are retained in host RAM. No summarizer and no
  silent history truncation. The checkpoint's native template decides whether
  old reasoning fields enter future model inputs.
- Requests rejected for context overflow stop that game and record the server
  error. Before a long evaluation, explicitly choose/test a context policy or
  a sufficiently supported context capacity. The runtime does not claim that
  unbounded full-history play fits within a finite context window.
- Malformed, missing, multiple, or truncated tool calls stop the game rather than
  guessing an action or starting an unbounded repair loop. This is conservative
  baseline behavior, not a claim that it is the strongest game policy.
- Transport failures are not automatically retried. A possibly executed action
  is never replayed after an uncertain outcome. The journal preserves the intent
  and the absence of an observed result. Official scoring remains authoritative.
- One journal per game is created exclusively. A duplicate attempt to own the
  same game in the same run directory fails rather than silently doubling work.
  This is not a crash-recovery or exactly-once network protocol.

`MAX_GAME_ACTIONS` and `MAX_DECISIONS` default to 10,000 as high safety limits;
`MAX_TOOL_ACTIONS` defaults to 32 primitive actions. Change these explicitly when
comparing policies; do not label changed action/reasoning budgets a serving-only
speed improvement.

## Why concurrent threads are sufficient

The official Swarm already creates independent game threads. A thread blocks
while its HTTP request is served; the other threads keep running. vLLM is the
shared asynchronous engine and continuously schedules the concurrent requests.
Threading here does not create multiple GPU models. An asyncio rewrite is not
required to unlock batching.

The small FIFO admission gate caps in-flight model requests. Its condition lock
is held only to update counters. Never hold a global model lock across the HTTP
request. Never wait for a whole group of game decisions to finish before executing
the first completed game's actions. The engine owns batching; the application
only submits ready work.

The stock Swarm creates all environments before starting their threads. This
adapter preserves that lifecycle and does not invent an unverified per-game
watchdog policy. With many games, measure queue waits and prefix eviction on the
actual evaluation-sized workload. Active GPU sequences, runnable game threads,
and total cached histories are different quantities.

## Benchmarking

Keep checkpoint, prompts, frame encoding, sampling parameters, output cap and
action policy fixed. Start with concurrency 1, 4, 8 and 16 on representative early
AND long-history requests. Use a higher engine max sequence cap than the largest
client concurrency under comparison. Do not run a large tuning sweep inside the
nine-hour scored run.

For a bounded development run, set `TRACE_FULL_REQUESTS=1`. This writes replayable
complete requests; leave it off for scored runs to avoid repeatedly logging large
historical prefixes. Normal journals retain the initial observation, all model
responses, action submissions/results and model request configuration, so the
conversation can be reconstructed without quadratic prompt logging.

```bash
python benchmark.py \
  --logs '/kaggle/working/one_tool_run_*/**/events.jsonl' \
  --concurrency 8 --limit 64 --output benchmark_c8.json
```

Use a fresh server with the same warmup and cache state for each comparison.
The command runs one concurrency point. It reports emitted token throughput and
valid completed tool replies per minute, including malformed/truncated replies
as failures. Actual sampled generations can differ with batching. A fixed prompt
replay does not reproduce closed-loop game arrival patterns, so confirm the
selected configuration with a small live run before freezing it.

For chunked prefill, compare `MAX_BATCHED_TOKENS=8192` and `16384` only after
choosing a useful concurrency. Smaller chunks can improve latency yet reduce
aggregate throughput. This runtime's target is completed useful decisions in the
fixed run, not pretty streaming latency.

Inspect vLLM metrics for running/waiting requests, prefix-cache hits, cache
occupancy and preemptions. `launch.py` saves `/metrics` before and after game play.
High GPU utilization alone is not success: it can coexist with recomputation,
malformed generations and increasing queue delay. Select the lowest concurrency
near the useful-throughput plateau with sufficient cache headroom at late-game
context lengths.

Quantized weights, FP8 KV cache, speculative decoding, shortened history, or a
smaller reasoning budget are separate experiments. Some affect numerical output
or available evidence; none is automatically faster or better. Do not enable all
at once and attribute the result to continuous batching.

## Source interfaces inspected

Official competition runtime and parallel games:
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data
https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Agents/main/agents/swarm.py
https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Agents/main/agents/agent.py
https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Agents/main/main.py
https://docs.arcprize.org/toolkit/competition_mode
https://github.com/arcprize/ARC-AGI

Serving and model-specific protocol:
https://docs.vllm.ai/en/stable/configuration/optimization/
https://docs.vllm.ai/en/stable/features/reasoning_outputs/
https://docs.vllm.ai/en/stable/features/tool_calling/
https://docs.vllm.ai/en/stable/examples/reasoning/openai_chat_completion_tool_calls_with_reasoning/
https://qwen.readthedocs.io/en/latest/deployment/vllm.html

Sources are interface references, not evidence that this generated bundle has
been run on Kaggle. No implementation from those projects is included in this
bundle; its code is an independently written adapter using those interfaces.
