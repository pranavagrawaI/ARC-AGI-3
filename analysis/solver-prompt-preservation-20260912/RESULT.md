# Result: solver prompt and tool-result preservation

Status: locally qualified; no gameplay-efficacy or live-run claim.

## Premise attacked

The failed shared premise was that improving the critic would cause level progress. Three critic designs reached the same zero-level gate. The next useful intervention is therefore not another critic policy but evidence that can distinguish an Actor belief from something the host actually rendered.

## Change

Created `deployment/kaggle/submission-runtime-20260912-v11` from the untouched v10 incumbent. Every Actor and critic `model_calls.jsonl` row now preserves:

- `request_messages`: the provider-prepared message list, including NOOA cache-control rendering and the serializer's omission of null dictionary fields;
- `tool_results`: API `role=tool` messages plus NOOA's role=user `PythonOutput(tool_call_id=...)` blocks.

The evidence is frozen before each request so later framework mutation cannot rewrite the record. Candidate identity is `submission-runtime-20260912-v11`, based on v10, with contract SHA-256 `864838a6acde0530e4e54d6cd74b607ad2046c35dcbdff3b15a222f6ad836c2a`.

## Discriminating test

The regression fixture traverses the real Actor-to-worker-to-host path. Its local HTTP server independently captures the JSON request body, then the test requires all three recorded `request_messages` lists to equal those bodies exactly. It also requires the second request's `tool_results` to contain the first cell's independently known `call1` output, `inspection`, and confirms that the host refuses the Actor's false DONE and still receives ACTION1.

Preimplementation result: failed at the new seam with `KeyError: 'request_messages'` (`local-v1`, 6.28s).

Final targeted result (`local-v6`):

    1 passed in 6.22s

Full frozen-candidate suite:

    175 passed in 64.63s (0:01:04)

Freeze check:

    864838a6acde0530e4e54d6cd74b607ad2046c35dcbdff3b15a222f6ad836c2a

External model calls: 0. ARC environment calls: 0. Spend: $0.

## Interpretation

The observability gap is closed for future v11 Actor and critic calls: a claimed host event can now be checked against the exact recorded request and its rendered tool outputs. This does not explain the prior LS20 `100/100` or `Context was evicted` beliefs, because those v9/v10 requests were not preserved, and it does not show that v11 solves more levels.

Runtime v10 remains the gameplay incumbent. V11 is the locally qualified observability successor. The next discriminating move is a separately carded LS20 replay or live contrast using v11 evidence; no Kaggle launch was made or authorized here.

## Run trail

- `EXPERIMENT-CARD.md` / `RESULT-v1.md`: red established; green invalidated by candidate drift.
- `EXPERIMENT-CARD-v2.md` / `RESULT-v2.md`: source messages differed from the HTTP boundary.
- `EXPERIMENT-CARD-v3.md` / `RESULT-v3.md`: cache-control preparation alone remained inexact.
- `EXPERIMENT-CARD-v4.md` / `DIFF-v4.json` / `RESULT-v4.md`: all 12 differences were null `content` fields omitted by transport serialization.
- `EXPERIMENT-CARD-v5.md` / `RESULT-v5.md`: exact request equality passed; the tool-result oracle exposed NOOA's two result renderings.
- `EXPERIMENT-CARD-v6.md`: final passing qualification.
