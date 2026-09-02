# R8-L3B — patched SGLang live lifecycle qualification candidate

**Preparation state:** The existing R8-L3B notebook path now contains candidate generation 4 for same-slug Kaggle version 14. The user authorized continuing manual iteration on that online notebook. Each frozen generation permits one manual push with zero automatic retries; no dataset update, S0, ARC gameplay, submission, or promotion is authorized.

Generation 1 was rejected before upload because it emitted `datasets-metadata.json`, while Kaggle CLI `2.2.4` consumes `dataset-metadata.json`. Its exact changed bytes and full dataset directory are preserved under `analysis/deployment-probes/20260831T130423Z-qwen38-r8l3b-candidate-local/rejected-generation-1/`; no review or authority transfers from it.

Generation 2 created private dataset version 1 successfully, but provider readback showed that Kaggle removed `+` from the wheel filename while preserving every wheel byte. Its notebook required the untransformed name and is rejected. Exact bytes are preserved under `analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/rejected-notebook-candidate-g2/`.

Generation 3 became immutable Kaggle version 13 and failed at `5.512s`, before installation, because it assumed `/kaggle/input/datasets/<owner>/<slug>` mount roots. Its source, log, verdict, and rejected local candidate are preserved under `analysis/deployment-probes/20260831T145806Z-qwen38-r8l3b-kaggle-v13/`. Generation 4 replaces only that physical-path assumption with bounded exact-name-and-hash discovery under `/kaggle/input`.

**Question:** Does replacing only the pinned SGLang wheel with the exact R8-L3B orderly-shutdown patch preserve the accepted request path while eliminating every forbidden teardown byte and retaining root-0, TERM-only complete process/listener/GPU closure on the bound Kaggle accelerator?

**Posture:** Lifecycle qualification. R8-B2 request behavior and the R8-L2 external observer are non-regression requirements. Diagnose only after a classified failure.

## Predecessor

Kaggle kernel version 12 is a valid classified failure: request path PASS; one runtime warmup plus one probe request; root return code `0`; TERM-only complete closure at `3.012s`; lifecycle `8.034s`; but strict teardown-log FAIL on the NCCL `destroy_process_group()` warning, async `SystemExit: 0` traceback, and Starlette/Uvicorn `CancelledError` traceback.

- verdict SHA-256: `0f9977f3f47d8149ff515d7e0c20997f2f93fa8990dcfbf49ff9b903d2cd0ded`;
- independent classification review SHA-256: `c241e30adfaa1367028fddb01b069f182447d5ca49f4b2461d37f9f9ceceba14`;
- consumed version-12 push grants no retry.

## Treatment and invariants

The only runtime treatment is the exact patched wheel:

- base SGLang commit: `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`;
- base wheel SHA-256: `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`;
- patched wheel SHA-256: `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0`;
- inherited lock SHA-256: `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`.

The patch changes shutdown ownership and ordering only: tokenizer shutdown requests ordinary Uvicorn exit and returns normally; the default single-tokenizer Uvicorn owner exposes `server.should_exit`; scheduler host/distributed cleanup requires a normal event-loop return plus graceful shutdown, with distributed cleanup in `finally` around admitted host cleanup.

Preserve unchanged:

- model dataset and exact model revision;
- inherited base runtime dataset and every dependency wheel other than SGLang;
- Triton attention, PyTorch sampling, reasoning parser, tool parser, CUDA graph/cache settings, and port `30000`;
- exact 2,447-byte request, SHA-256 `f4108f8f6b05f32bd84dd4bfaad69968cff537d93a09d467d0f77393a6413cf0`;
- one successful runtime warmup plus one successful probe request;
- R8-L2 root-only TERM treatment, external ownership observer, PID/start-time identity, containment, deadlines, quiet-window rule, and default-deny teardown taxonomy;
- startup/request/session timeouts `1200/180/1800` seconds;
- zero ARC actions, zero submissions, and zero automatic retries.

## Thin runtime dataset

Private dataset version 1 carries only the patched SGLang wheel, the inherited lock, deterministic metadata/manifest, and exact local provenance. All remaining dependencies still resolve from the separately attached, unchanged base runtime wheelhouse. The generated notebook searches only `/kaggle/input` for the exact expected filenames and hashes, requires one unique patched wheel, patch manifest, base wheel, and non-patch inherited lock, and requires the patch wheel/manifest/lock to be co-located. It copies the patched wheel into `/kaggle/working` under the canonical `+` filename, verifies the copy, then gives the resolver only the inherited wheelhouse while passing the canonical copied wheel as an explicit path.

This arrangement is deliberate: copying the multi-gigabyte dependency wheelhouse would create an unreviewed substrate replacement; using only the thin dataset as `--find-links` would not be offline-installable.

## Frozen local candidate

Machine-readable candidate manifest:

- `deployment/kaggle/r8l3b-candidate.json` — SHA-256 `663a30f0bec48721f23d259d8a75b504b92a083af23182aa311c1844671e27f5`.

Dataset participants:

| Participant | SHA-256 |
|---|---|
| `deployment/kaggle/scripts/build_r8l3b_runtime_dataset.py` | `1766e4c97d24f87eefc848aa0c999c1ff10ebd9c936476cfac841d14c9714d6f` |
| `deployment/kaggle/r8l3b-runtime-dataset/dataset-metadata.json` | `aea8f265d1ed127aa0c21a244c3130840d1f4d767cda7c9799034ed527bed3f2` |
| `deployment/kaggle/r8l3b-runtime-dataset/runtime-dataset-manifest.json` | `f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94` |
| `analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/REMOTE-READBACK.json` | `cee470f08217da591b8a89d3a80f1f9b4f03aecffced29f69d1b2a5c510fda9d` |
| `analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/PROVIDER-STATE.json` | `55aef66ebfc43d1cd6ae0d159bc5049b96cb9c2ad8b205ff2891abe607a8237d` |
| `analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/ARTIFACTS.sha256` | `fe479b2ec428fbdf4547d83342ec3463dd3063cb19ff624fa988e9804139f450` |

Notebook participants:

| Participant | SHA-256 |
|---|---|
| `deployment/kaggle/PROTOTYPE-qwen3_8_r8_l3b_probe.py` | `b792325c746a76440703fda793252159364ba256251e939d0039cc6c1e8ed64f` |
| `deployment/kaggle/scripts/build_r8l3b_probe_notebook.py` | `7c2025c8cf2a63798a004f93facd8d8e32f9d8f79e0bb6599887ef17585fc84e` |
| `deployment/kaggle/r8l3b-notebook/r8l3b_probe.ipynb` | `f8d5241c12acd6f01d93e35d07aa0c4f3bb163e1dd74736fbb95f9ad4ec44afb` |
| `deployment/kaggle/r8l3b-notebook/kernel-metadata.json` | `a5fb2aa06ad500fa723e8302368ea74084d31033a3b9e0482226b2ca530ce93c` |
| `deployment/kaggle/tests/test_r8l3b_candidate.py` | `138ca5b2c5d81f5696cd6a3bba6a6688d398b3a29a20974685469a56d0726fc7` |
| `deployment/kaggle/r8l3b-launch.json` | `2f1e977996f5aecf97c1940d78b5bb51abcfcecc4178e3eeae3efc62cc85ecb3` |

Any byte change creates a new candidate and invalidates reviews of these hashes.

## Local acceptance requirements

Before either external boundary:

1. recompute every machine-readable participant and every uploaded dataset file;
2. regenerate the dataset twice and require byte identity with the frozen directory;
3. regenerate notebook/metadata and require byte identity;
4. compile every generated code cell and decode exact embedded probe/model-manifest bytes;
5. execute the generated content-addressed mount-selection/install projection under a provider-shaped slug layout, require a byte-identical canonical copy, and require the untransformed remote name, manifest drift, and duplicate exact patch wheel to fail before install;
6. preserve request-function AST identity and exact request bytes;
7. run the complete deployment suite and Ruff;
8. bind the exact candidate and one-push launch bytes to the user's continuing same-slug iteration authority.

Local success licenses only a kernel-launch authorization request against the already-read-back private dataset version 1.

## External authority boundaries

### Boundary 1 — dataset upload — consumed

One authorized create attempt produced private dataset version 1 with zero retry. All payload bytes read back exactly; Kaggle normalized the wheel filename. Evidence is frozen at `analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/`. No dataset update is authorized or required by generation 4.

Generation 4 consumes the observed normalized name and version-1 readback directly. It does not reinterpret the generation-2 upload as an unchanged-filename success.

### Boundary 2 — kernel push and GPU run

The prepared launch command is the exact argv in `deployment/kaggle/r8l3b-launch.json`, bound to private dataset version `1`, its normalized remote wheel name and readback hash, remote predecessor version `13`, expected successor `14`, Kaggle CLI `2.2.4`, RTX PRO 6000, private/offline metadata, one manual push, zero automatic retries, and maximum `0.50` GPU-hours. The user's continuing same-slug iteration instruction supplies launch authority after fresh remote version/status/quota preflight.

Dataset authorization never implies kernel authorization.

## Live result map

- attached bytes, privacy, internet, accelerator, predecessor, or version differ → `R8-L3B INVALID`; no model interpretation;
- offline installation cannot prove the explicit patched wheel plus unchanged inherited dependency wheelhouse → `R8-L3B INVALID`; preserve evidence, no retry;
- request or parser/count behavior differs → `R8-L3B FAIL: request_non_regression`; lifecycle result cannot rescue it;
- root-only TERM requires KILL or leaves process/listener/GPU residue → retain the existing R8-L2 closure claim, reject R8-L3B live lifecycle;
- any forbidden or unclassified teardown byte remains → `R8-L3B FAIL: teardown_log_unacceptable`;
- exact request PASS plus TERM-only root-0 complete resource closure plus strict zero-forbidden-byte log acceptance → `R8-L3B PASS`.

A PASS may authorize only preparation of a separately frozen one-cold-action S0 candidate. It never authorizes S0 execution automatically.

## Stop

Never automatically retry the same candidate. Stop the current generation when any participant hash drifts, private dataset version 1 differs from readback, the predecessor/successor pair differs from `13 → 14`, quota or accelerator cannot be established, request behavior changes, any observer fact is unavailable, KILL is required, or any teardown byte is forbidden/unclassified. Preserve the terminal generation before any manual successor edit.
