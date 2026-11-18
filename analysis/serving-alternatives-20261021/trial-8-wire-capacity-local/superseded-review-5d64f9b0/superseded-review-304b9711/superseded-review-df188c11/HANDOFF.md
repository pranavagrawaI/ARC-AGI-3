# Trial8 review handoff — LOCAL, UNASSIGNED

Review `EXPERIMENT-CARD.md`, `packet-spec.json`, `capacity-contract.json`, `api-source-review.json`, `causal-model.json` and the current `review-packet.json`. `review-packet-draft*` is retained obsolete preparation, not launch material.

New Kaggle identity: `pranavagrawai/serving-wire-capacity8-20261023`. Proposed mandatory platform10200s, cooperative supervisor9900s, one push/two loads/two fixed1500s replays/S10/at most2738 corpus requests. No real slot/assignment/attempt exists. This is a new contract, not trial4 qualification.

Reproduce local checks, from this directory:

```bash
python3 -m unittest discover -s tests -v
/home/pranav/.local/share/uv/tools/kaggle/bin/python -c 'import json,verify; print(json.dumps(verify.qualify(),indent=2))'
python3 -c 'import json,hashlib,pathlib; p=pathlib.Path("."); f=json.loads((p/"freeze.json").read_text()); assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==s for n,s in f["files"].items()); print(len(f["files"]),hashlib.sha256((p/"freeze.json").read_bytes()).hexdigest())'
```

The qualified future raw reducer command is `python3 reduce_api.py /path/to/collected/output`. It expects sibling `config.json`, supervisor/source/model/release receipts and both raw arm directories. Missing receipts yield FEASIBILITY_ONLY, not a ratio. To inspect one arm without asserting a qualified pair: `python3 reduce_api.py /path/to/output --arm control` (or candidate). Use copies of collected artifacts if avoiding creation of the local reduction JSON beside raw evidence.

All-corpus CPU integrity command, writing a NEW local output directory:

```bash
python3 replay.py prepare ../../mtp-depth-20261006/kaggle/dataset /tmp/trial8-unique-corpus-check
```

The stored `qualification/full-corpus.json` and `qualification/wire-bodies/index.json` bind the existing local check; these prepared body files are derived, not embedded in the packet. The future remote prepare repeats manifest/file/wire/PNG verification from the mounted pinned corpus. Catalog rebuild is `catalog.build()`; do not rewrite frozen outputs to reproduce it.

Remaining review/runtime boundaries: final exact source/field-path/card/cap review; explicit new trial8 reconciliation-policy review; live quota/session refresh and a new exact serialized assignment only if the coordinator chooses to authorize. Runtime must establish full-reply S10 fit, actual control usage/image totals, installed candidate processor hash and all predeclared API/load/stability/mix/length/source gates. No native residence/per-stream metric; no processor/tensor equivalence, engine-only gain, quality or adoption claim.

The launch entry point is intentionally fail-closed without new authority. Do not execute `launch.py` during local review. No inherited trial7 authority is valid. Any changed packet/card/cap after execution begins requires a separately identified successor; no retries.
