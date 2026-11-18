# Release11 bootstrap context diagnosis — LOCAL ONLY

Observed failure: the notebook's runpy.run_path(supervisor.py) does not put the
extracted package directory onto sys.path. The new supervisor alias imports the
sibling release_runner, but notebook sys.path does not include that directory.
The frozen failure therefore occurred before Runner/hardware/query/model/release.
This is source/receipt-supported; it is not nvidia incompatibility or release data.

The exact armed OUTER notebook cell and its exact decoded INNER cell both reproduce
ModuleNotFoundError release_runner under isolated Python from a foreign cwd with
clean stdlib-only sys.path. INPUTS.json binds original sources, failure receipts,
armed bytes and coordinator terminal closure. All166 original freeze bindings
rechecked, never edited. Remote attempt remains closed; no retry/assignment.

Proposed minimum correction in bootstrap_patch.py: prepend the extracted root to
sys.path, reject already-loaded payload module origin/hash mismatches, import the
real packaged release_runner and check loaded payload module origins/hashes before
calling the unchanged runpy supervisor entrypoint. Only this bootstrap preamble
is inserted; payloads, config, assignment, caps, science and all runtime files are
unchanged. This is a source transformation, NOT a newly prepared GPU packet.

Seven source-bound local cases passed in0.304s:

- Exact armed outer / clean foreign context reproduces original import error.
- Exact decoded inner / same context reproduces the same error.
- Corrected bootstrap enters real packaged main/Runner and reaches a CPU hardware
  phase boundary, then deliberately aborts NOT_QUALIFIED before actual hardware.
- Baseline with a preloaded wrong entry module calls the wrong main (safe marker).
- Corrected bootstrap rejects that preloaded entry module before its main.
- Corrected bootstrap rejects a wrong preloaded adapters dependency.
- Corrected bootstrap selects the real package over a foreign-directory shadow.

Both corrected positive cases use the actual extracted release_runner bytes and
real CLI/config/assignment parsing, source-pin imports and Runner construction.
The ONLY phase substitution stops hardware_and_queries with an explicit CPU-only
error; actual result/trace/cleanup show no owned groups/queries. This is entrypoint
wiring evidence, not hardware, model loading, health200 or release qualification.
All22 payloads are exact in every extracted case. No GPU/network/API/subprocess
or signal-dispatch attempts were made; audit guards forbid them.

Safe replay fixtures are explicit: pathlib root transposition into a case directory
and historical clock replay let exact failed source/assignment bytes execute only
inside the guarded local child. They grant no fresh authority. Isolated child
uses -I -B; every case's foreign cwd/sys.path/extracted bytes/result are retained.
Guard addresses ordinary shadowing and preloaded modules; it does not attest live
code objects or protect against hostile custom import hooks. Actual Kaggle bootstrap
execution remains unproved until a separately reviewed future experiment.

Reproduce, CPU only, fresh output:

```bash
timeout 600s python3 verify.py --output independent-FRESH
```

Proof proof-r2/PROOF.json SHA `cd88aba146599eedcbb040b91163510ab7b9ba8b03761b4a095f7918c5ca1f69`.
Patch bootstrap_patch.py SHA `e2db76e9011c12171e073cd49d8f95dde3a74ecc573bc66619775b31231d629d`.
All source files and case bytes are inventoried by that proof. Initial proof-r1/
failed in the Python3.14 Path-subclass test fixture before extraction, not the
runtime; retained. Fixture now uses a concrete PosixPath factory, with no change
to the proposed correction. No packet construction, GPU authority, launch, API,
threshold/cache/filter change, staging or commit occurred in this diagnosis.
