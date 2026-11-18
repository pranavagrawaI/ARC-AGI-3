# Validator → schema and feedback trace

Authority: previous reviewed `engine/constructor.py::_validate_work_item`, copied read-only from `../openrouter-constructor-reasoning-20260916T185550Z/`. This task changes interface schema and diagnostic feedback, not accepted work-item meaning, source handling or constructor policy.

| Existing rule | Emitted nested schema | Targeted feedback |
|---|---|---|
| value must be dict with exactly nine keys | object, required nine names, additionalProperties false | arguments.work_item expected object; received JSON type; exact missing/extra names |
| decision, obstacle, claim, remaining_scope must be nonblank str | string, minLength 1, pattern `\S` | field path + string/non-whitespace requirement |
| operation in ground/contrast/close/compose/test | string enum with these five members | field type or permitted choices |
| reasoning_need in needs_computation/needs_observation/does_not_block_decision | string enum with these three members | field type or permitted choices |
| evidence_refs and completion_checks nonempty list, integer entries in 1..frontier | array minItems 1; item integer minimum 1, maximum actual case frontier | indexed path + expected integer or visible range |
| dependencies nonempty list of nonblank strings | array minItems 1, string items minLength 1 / pattern `\S` | list or indexed string/nonblank error |

The complete required order is decision, obstacle, operation, claim, evidence_refs, dependencies, completion_checks, remaining_scope, reasoning_need. Enums/required names are shared constants in the copied constructor so emitted schema and diagnostic checks cannot drift independently.

The submit_candidate schema is filled with the actual case.frontier before dispatch (1 in this experiment). It is offered only after successful inspection, preserving the original policy. Tool count/names, tool_choice, candidate_source oneOf/text wording and maximum source handling are unchanged.

## Nesting and coercion

Missing work_item gives an explicit arguments.work_item path and instruction to nest the fields beside candidate_source. String (including JSON-encoded object text), null and array inputs are rejected with their received type. Empty, incomplete, extra-key and double-wrapped objects identify missing/extra keys. No JSON string is decoded, flattened input repaired, or field synthesized. Original malformed provider arguments remain byte-preserved; the model must submit a new valid call through normal budgets.

Feedback is only generated after the current episode makes an invalid call. No previous experiment error, source solution or game-semantic diagnosis is added to prompts.

## Preserved validator edge

The existing Python `isinstance(item, int)` check accepts boolean true as integer step 1 (false fails the positive bound). The schema explicitly includes `const: true` alongside the integer item form. This prevents silently narrowing a legacy acceptance boundary. A real HTTP/constructor/ABI regression verifies true remains unchanged in the accepted work_item. It is not a new recommended model representation.

JSON Schema integer describes mathematical integrality; it does not constrain number spelling on the wire (e.g. 1 versus 1.0). The existing JSON parser/host check remains authoritative and performs no coercion; float values continue to be rejected. No additional policy change is hidden in this interface repair.

## Decisive integration evidence

The new string regression ran against the copied unmodified constructor first and failed because feedback was only `work_item must contain the exact required fields`. `tests-before.txt` preserves that reproduction.

Each invalid-type/key/nesting regression then follows actual loopback HTTP → constructor → sealed inspection → invalid submission → exact feedback in the NEXT outbound HTTP request → valid nested object → real sealed initialize/predict/assimilate → ABI-only freeze. Raw malformed arguments are checked unchanged. Separate on/off executions compare the complete exposed nested schemas and use an independent JSON Schema validator. No helper-only success claim or paid request.
