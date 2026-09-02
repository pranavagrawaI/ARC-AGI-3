# Edsger Dijkstra — fresh exact-byte R8-L3B artifact/protocol review

## Decision boundary

**ACCEPT for local artifact closure only.** The five frozen participants form a deterministic, installable wheel transformation from the pinned upstream bytes. This review does not establish SGLang/Qwen startup, request-path preservation, GPU or NCCL behavior, clean Kaggle teardown, or any live result. It grants no upload, push, GPU execution, retry, ARC action, S0 preparation or run, submission, or promotion authority.

## Frozen identity

The five supplied hashes matched before substantive inspection and again after all builds, tests, install checks, archive checks, and negative controls. I did not edit these participants.

| Participant | SHA-256 before | SHA-256 after |
|---|---|---|
| `deployment/kaggle/PROTOTYPE-r8l3b-sglang-orderly-shutdown-EXPERIMENT-CARD.md` | `ca77e4a1a4bf49c3a32b007ab7b36f0509d79d1e1729a2dffd22691c378ec4aa` | `ca77e4a1a4bf49c3a32b007ab7b36f0509d79d1e1729a2dffd22691c378ec4aa` |
| `deployment/kaggle/scripts/build_sglang_r8l3_wheel.py` | `3ce16555e2625ef72d10bc4e9096d8eed320a9be8f6ea33aa539505900220c6b` | `3ce16555e2625ef72d10bc4e9096d8eed320a9be8f6ea33aa539505900220c6b` |
| `deployment/kaggle/tests/test_sglang_r8l3_wheel.py` | `43f90de8c977248deb7eb8eb7913725437177075aa618eab86562b14f1da04ee` | `43f90de8c977248deb7eb8eb7913725437177075aa618eab86562b14f1da04ee` |
| `deployment/kaggle/runtime-patches/r8l3b/manifest.json` | `adb11a6608be04399bd903de99c96288ace6d69fd58520214de3dc9c98c2a630` | `adb11a6608be04399bd903de99c96288ace6d69fd58520214de3dc9c98c2a630` |
| `deployment/kaggle/runtime-patches/r8l3b/sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl` | `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0` | `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0` |

Repository `HEAD` during review was `88108520f0294530feb481f795cf4d3ea0d37111`; all five participants were untracked, so the requester's supplied hashes—not Git—are their external freeze.

## Exact upstream and reproducible build

The upstream wheel at `/tmp/sglang-r8l3-runtime/sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl` recomputed to the required SHA-256 `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`. Its companion `requirements.lock` recomputed to `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`, matching the manifest.

The source checkout at `/tmp/sglang-r8l3-source` was clean at commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`. Each of the three upstream wheel source entries was byte-equal to the corresponding checkout file.

Two fresh builder invocations from that exact upstream wheel produced byte-identical outputs. Both outputs were byte-equal to the frozen candidate and had SHA-256 `06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0`.

## Wheel delta and RECORD closure

The upstream and candidate archives contain the same 4,266 unique member names in the same order. Exactly four entries differ:

| Entry | Upstream SHA-256 | Candidate SHA-256 |
|---|---|---|
| `sglang/srt/entrypoints/http_server.py` | `056422af61f51573256560509d42ed4f23151364158474aff1c012cce95c880b` | `c47245e4f64744301dcd9669d3e76984a918b0d8a50b0d0611d8b4dc81ffddbb` |
| `sglang/srt/managers/scheduler.py` | `0d4190dcff3dcda16812fbe95f08ec9c04234a8f12e80a81ff1406c58d46f748` | `dbb04590d727493498ea537bb1b70dc57d9b636da3ed78361093277e6d244b77` |
| `sglang/srt/managers/tokenizer_manager.py` | `0d194294e7d56fe71cb7f13701d6021a0f7c41f5b6480319627ee1bbe8d94110` | `90bf218aec5cf24be7c822821bca6527a3bb1dc834824eae81a06ea862c30cdf` |
| `sglang-0.5.19.dev591+g8e04f66a7.dist-info/RECORD` | regenerated | regenerated |

For the other 4,262 entries, uncompressed bytes, raw compressed payloads, CRCs, sizes, compression methods, timestamps, attributes, comments, extras, and relevant ZIP metadata matched. The three changed source entries retained their upstream ZIP metadata apart from content-derived CRC and sizes. No hidden entry or metadata change survived comparison.

The candidate has exactly one `RECORD`. I parsed all 4,266 rows and required exactly three fields, one unique row per archive member, and archive-order equality. Every non-`RECORD` row's URL-safe base64 SHA-256 and decimal byte length recomputed exactly. The sole `RECORD` row had the required empty digest and size. There were no missing, duplicate, extra, malformed, or stale rows.

## Source projection and treatment honesty

Applying `patch_runtime_sources` independently to the three exact upstream entries produced bytes equal to the three candidate entries. All six upstream/patched source hashes match the manifest's `source_projection`, and all three patched files parse as Python.

The implemented projection matches the card and manifest:

1. tokenizer shutdown retains watchdog stop, `ShutdownReq` dispatch, scheduler wait, and child containment order; it then requires and invokes the HTTP-server shutdown callback instead of raising async `SystemExit(0)`;
2. the frozen default single-tokenizer HTTP branch constructs and owns `uvicorn.Server`, binds a callback that sets `server.should_exit = True`, calls `server.run()`, and clears the callback in `finally`;
3. `scheduler_loop_returned_normally` starts false and becomes true only after `scheduler.run_event_loop()` returns;
4. host and distributed cleanup require both `scheduler.gracefully_exit` and that normal-return witness; and
5. `cleanup_dist_env_and_memory()` is in `finally` around host cleanup, so it still runs if host cleanup raises on the admitted normal-return path.

The projection is deliberately narrower than wheel-wide generic shutdown support: the callback binding is for the frozen default single-tokenizer Uvicorn route. The card frames this as a later-qualification treatment, not as locally proven live behavior.

## Installation and executable discrimination

A fresh `/usr/bin/python3` 3.12 virtual environment installed the candidate with `pip --no-deps`. Installed metadata reported `0.5.19.dev591+g8e04f66a7` from that environment's `site-packages`. `compileall` passed over the installed `sglang` package; it emitted the one unrelated pre-existing invalid-escape `SyntaxWarning` already disclosed by local verification. A no-dependency environment is not a full runtime-import qualification.

Canonical checks reproduced:

- `uv run --project deployment/kaggle --frozen pytest -q deployment/kaggle/tests` → `61 passed, 1 skipped`;
- `uv run --project deployment/kaggle --frozen ruff check deployment/kaggle/scripts/build_sglang_r8l3_wheel.py deployment/kaggle/tests/test_sglang_r8l3_wheel.py` → passed;
- the two focused behavioral discriminators → `2 passed`.

The watchdog discriminator executes the old fixture and requires `SystemExit`, then executes the transformed fixture and requires normal return after the ordered `watchdog_stop`, `dispatch_shutdown`, `kill_children`, and `server_exit` events. The scheduler discriminator sets `gracefully_exit=True`: normal event-loop return must perform metrics, host, and distributed cleanup, while an exception after the shutdown condition must perform metrics cleanup only. This directly distinguishes the rejected predecessor's defect; it is modeled local execution, not SGLang/GPU qualification.

## Negative controls

Disposable controls, without editing any participant, failed closed:

- a one-byte-wrong whole upstream wheel was rejected by the complete upstream-wheel SHA-256 gate before output creation;
- each of the three source entries was separately mutated in a disposable copied wheel; after rebinding only the disposable outer-wheel digest to reach the inner gate, the builder rejected the wrong source-entry SHA-256 before output creation.

Thus the outer identity gate and all three source-projection gates were exercised separately.

## Manifest, card, and predecessor bindings

The manifest's builder, tests, and wheel participant hashes recomputed from their exact paths. The externally supplied freeze binds the card and manifest themselves, avoiding a recursive self-hash claim. The card names the same candidate wheel digest, upstream wheel digest, predecessor hashes, treatment, and authority boundary as the manifest.

All four predecessor references resolve and recompute exactly:

| Binding | Referent | SHA-256 |
|---|---|---|
| rejected Kaggle v12 verdict | `analysis/deployment-probes/20260830T212134Z-qwen38-r8l2-kaggle-v12/VERDICT.md` | `0f9977f3f47d8149ff515d7e0c20997f2f93fa8990dcfbf49ff9b903d2cd0ded` |
| independent review of v12 classification | `analysis/deployment-probes/20260830T212134Z-qwen38-r8l2-kaggle-v12/POST-RUN-INDEPENDENT-REVIEW.md` | `c241e30adfaa1367028fddb01b069f182447d5ca49f4b2461d37f9f9ceceba14` |
| rejected R8-L3 implementation review | `analysis/deployment-probes/20260831T102619Z-sglang-r8l3-local/IMPLEMENTATION-REVIEW.md` | `b45cd965c20df76a7fa369f363d0f3475166944c627e54109565c039678f4bb8` |
| packaging-only R8-L3 artifact review | `analysis/deployment-probes/20260831T102619Z-sglang-r8l3-local/ARTIFACT-REVIEW.md` | `c104d1008894e9ac1e10d5e7c0ab22f2e17c8c6b6a281869ab6d15bf65d32741` |

The v12 verdict is `REJECT: teardown_log_unacceptable`; its independent review accepts that rejection classification, not the run. The first local R8-L3 implementation review rejects mutable-state-only cleanup gating, while its separate artifact review accepts packaging only. R8-L3B adds the demanded normal-return witness and a discriminating exception-after-shutdown control. None of these predecessor references authenticates live success for the new wheel.

## Authority ceiling

The card says the candidate is not live-qualified or launchable. The manifest status is `local_artifact_built_not_live_qualified`, and its claim ceiling explicitly excludes GPU startup, request non-regression, distributed teardown, and clean Kaggle lifecycle. Predictions are labeled as predictions for later qualification. I found no hidden extra wheel change and no local-to-live claim.

Acceptance therefore licenses only retention of these exact local artifact bytes. Dataset construction or upload, notebook binding, Kaggle push, GPU execution, retry, ARC action, S0 preparation or execution, gameplay, submission, and promotion all remain outside this review and require a separately frozen candidate plus separate user authority.

VERDICT: ACCEPT