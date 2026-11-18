# Trial8 review handoff — LOCAL, UNASSIGNED

Review `EXPERIMENT-CARD.md`, `packet-spec.json`, `capacity-contract.json`, `api-source-review.json`, `causal-model.json` and the current `review-packet.json`. Current binding is `freeze-r3.json`. Both `superseded-review-df188c11/` and `superseded-review-304b9711/` retain byte-exact obsolete review preparation, not launch material. `verify.qualify()` checks local files under each snapshot and external bindings from this trial8 directory.

Only `review-packet/` and `review-packet.json` are current; other `review-packet-*` paths are historical. Current transport proof is `qualification/sdk-lossless-r3.json`, full suite `qualification/tests-r3-final-v2.log`, and exact new repair evidence `diagnosis/protocol-r3/REPAIR-RECEIPT.json`. Earlier persistence repair evidence remains under `diagnosis/persistence-r2/`.

New Kaggle identity: `pranavagrawai/serving-wire-capacity8-20261023`. Proposed mandatory platform10200s, cooperative supervisor9900s, one push/two loads/two fixed1500s replays/S10/at most2738 corpus requests. No real slot/assignment/attempt exists. This is a new contract, not trial4 qualification.

Reproduce local checks, from this directory:

```bash
python3 -m unittest discover -s tests -v
/home/pranav/.local/share/uv/tools/kaggle/bin/python -c 'import json,verify; print(json.dumps(verify.qualify(),indent=2))'
python3 -c 'import json,hashlib,pathlib; p=pathlib.Path("."); f=json.loads((p/"freeze-r3.json").read_text()); assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==s for n,s in f["files"].items()); print(len(f["files"]),hashlib.sha256((p/"freeze-r3.json").read_bytes()).hexdigest())'
```

The qualified future raw reducer command is `python3 reduce_api.py /path/to/collected/output`. It expects sibling `config.json`, supervisor/source/model/release receipts and both raw arm directories. Missing receipts yield FEASIBILITY_ONLY, not a ratio. To inspect one arm without asserting a qualified pair: `python3 reduce_api.py /path/to/output --arm control` (or candidate). Use copies of collected artifacts if avoiding creation of the local reduction JSON beside raw evidence.

All-corpus CPU integrity command, writing a NEW local output directory:

```bash
python3 replay.py prepare ../../mtp-depth-20261006/kaggle/dataset /tmp/trial8-unique-corpus-check
```

The stored `qualification/full-corpus.json` and `qualification/wire-bodies/index.json` bind the existing local check; these prepared body files are derived, not embedded in the packet. The future remote prepare repeats manifest/file/wire/PNG verification from the mounted pinned corpus. Catalog rebuild is `catalog.build()`; do not rewrite frozen outputs to reproduce it.

Exact persistence/cutoff repair probe, each writing a NEW output directory:

```bash
python3 diagnosis/persistence-r2/reproduce.py superseded-review-df188c11 /tmp/trial8-r2-before-unique
python3 diagnosis/persistence-r2/reproduce.py . /tmp/trial8-r2-after-unique
```

Stored before/after primary raw receipts and analytical false-verdict witness are under `diagnosis/persistence-r2/`. These use CPU loopback and shaped raw fixtures, not model performance. The former source incorrectly reports successful censoring and a synthetic qualified ratio despite disk loss; the repair reports failure/FEASIBILITY_ONLY while accepting ordinary intentional cutoff. `qualification/*persistence*` preserves red/green logs. All original sources, packet and freeze are retained in the superseded snapshot.

Exact protocol/cutoff repair probe, with actual malformed status-line parsing and explicit delayed exception delivery:

```bash
python3 diagnosis/protocol-r3/reproduce.py superseded-review-304b9711 /tmp/trial8-r3-before-unique
python3 diagnosis/protocol-r3/reproduce.py . /tmp/trial8-r3-after-unique
```

This is scheduling fault injection, not an estimate of error incidence. r2 incorrectly reports success and accepts parser-error censoring. Current source fails and rejects it; actual owned-shutdown RemoteDisconnected/ordinary trickle remain accepted. Raw receipts, inventories and red/green evidence are under `diagnosis/protocol-r3/` and `qualification/protocol-r3-*`.

Remaining review/runtime boundaries: final exact source/field-path/card/cap review; explicit new trial8 reconciliation-policy review; live quota/session refresh and a new exact serialized assignment only if the coordinator chooses to authorize. Runtime must establish full-reply S10 fit, actual control usage/image totals, installed candidate processor hash and all predeclared API/load/stability/mix/length/source gates. No native residence/per-stream metric; no processor/tensor equivalence, engine-only gain, quality or adoption claim.

The launch entry point is intentionally fail-closed without new authority. Do not execute `launch.py` during local review. No inherited trial7 authority is valid. Any changed packet/card/cap after execution begins requires a separately identified successor; no retries.
