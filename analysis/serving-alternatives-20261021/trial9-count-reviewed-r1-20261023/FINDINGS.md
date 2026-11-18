The prompt reference was evaluated before the replay's JSON serialization
boundary. The replay serializes with `catalog.canonical`, which sorts all nested
object keys. The pinned SGLang Tool/Function serializers preserve the insertion
order inside `function.parameters`; the template serializes those parameters
into model input. JSON semantic equality therefore does not establish equality
of rendered prompt bytes or token IDs.

Six parameters-only counterfactuals isolate this mechanism: changing only the
actual request's parameters dictionary back to source insertion order restores
the old catalog render SHA and count. The reference and actual request have the
same canonical semantic wire hash and full source PNG hashes throughout. The
regression tests cover this distinction and reject reordered or whitespace-
changed HTTP body bytes under the new exact prepared-body SHA binding.

The corrected reference is separately identified in `catalog.exact-wire-v2.json`.
It is derived from each exact serialized request body, parsed through the pinned
control and candidate tool classes, rendered through the control's pinned loader
and candidate normalized template, and encoded with the corresponding local
model tokenizer. It does not subtract an empirically observed constant. The
recorded new-minus-old histogram describes the result only. Immutable source
wire hashes, images, FIFO stream order and completion caps remain unchanged.

`ALL-CORPUS-PROOF.json` and `exact-wire-records.jsonl` bind all1369 local reference
records, source versions, actual serialized body hashes, render and token-ID
hashes. `verify_snapshot.py` additionally verifies the original642 trial9 inputs,
pinned wheel/loader/protocol source bytes, and recovered members used in the275
raw-HTTP/body joins. The original trial9 catalog and accounting gate are not
changed; trial9 remains FEASIBILITY_ONLY with no ratio.

For275 recoverable responses, actual HTTP body bytes match the reference bytes;
raw API prompt totals match the actual-wire CPU count. All275 image aggregates
match400 times their source PNG counts. These include264 cache-hit and11
zero-cache responses. Complete usage arithmetic, completion-cap and finish
fields also pass their independent checks. The original prompt gate still
rejects all275 against its old reference. Missing266 completed joins remain
missing; archive-order selection is not representative sampling.

Trial7's six actual diagnostic requests used thinking=false, forced python,
256-token reply caps and a diagnostic instruction. CPU rendering of those exact
bytes reproduces the observed candidate prompt totals. Switching only
thinking=true changes their CPU counts by36 tokens. Those six diagnostic
responses do not establish production-thinking/runtime equivalence. In the six
trial9 setting counterfactuals, template-loader preprocessing and text
encode/decode/reencode do not explain the one-token discrepancy; restoring
parameter order does.

Image accounting remains distinct from prompt reference construction: source
PNG hashes/counts come from actual HTTP input,400 tokens per image is supported
by pinned configuration and observed aggregate API totals, while pixel/tensor
processing and each-image-once correspondence remain unproved. Both CPU frontend
text references agreeing does not establish actual candidate production
processor behavior. No native input tensors, image correspondence, quality,
S10-fit generalization or rate/adoption verdict is inferred.

Reproduce locally (offline cached CPU dependencies,600s all-corpus cap):

```sh
uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pydantic==2.14.1 python -B analysis/serving-alternatives-20261021/trial9-count-local-20261023/all_corpus.py
uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pydantic==2.14.1 python -B -m unittest discover -s analysis/serving-alternatives-20261021/trial9-count-local-20261023 -p test_reference.py -v
python3 analysis/serving-alternatives-20261021/trial9-count-local-20261023/verify_snapshot.py
```

All-corpus reruns regenerate diagnostic outputs; preserve this audit snapshot
before running them. For inspection without changes, use only verify_snapshot.
The snapshot is an audit receipt, not a GPU preparation freeze. Successor packet
preparation stays on hold for coordinator judgment; no assignment or launch,
credential changes, archive retry, staging or commits occurred.

The separately accepted cutoff correction remains unchanged, including its
retained limitation: a pre-cutoff EOF whose handler is delayed past the owned
shutdown marker can still be classified as intentional cutoff. Native
TimeoutError is always fatal and definite protocol/persistence protections were
verified, but no blanket exclusion of all predeadline transport failures is
claimed.
