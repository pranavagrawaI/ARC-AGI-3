# Transport correction r1 — LOCAL REVIEW ONLY

Identity: `feas5-lossless-transport-proposal-r1-20261023`.
Status: unassigned; no launch script, SDK push, dataset upload, or remote mutation. Requested GPU cap NOW: zero. The failed feasibility assignment is closed and cannot authorize this proposal.

## Question and evidence boundary

Can the complete previously reviewed unassigned payload be represented as a smaller notebook with exactly identical decoded bytes and its original pre-extraction authority refusal? This is a CPU transport question, not a serving experiment. Notebook size is a plausible explanation for the SaveKernel400, not an established current platform limit or the recovered server rejection reason. Original SaveKernel response body/headers are unavailable.

The failed armed payload, launcher, all56 frozen preparation files, and redacted private stub are preserved. Status API twice reports404 “No runs found for this kernel.” GetKernel403 leaves version/source/existence unresolved; an exact-name search returns a redacted private stub. Final observed quota06:15:56Z: used59265.958s, reserved0, delta0. Coordinator explicitly closed assignment and released slot.

## Minimal change

`transport.py` wraps exact `packet-r2/serving-candidate-feas5-20261023.ipynb` bytes using Python stdlib LZMA preset6 and base64. The small outer notebook checks the decoded notebook SHA256 and exact source SHA256, then executes its single code cell. All inner sources, cases, full images, config, mounts, scientific gates, and assignment=None remain identical. `transport-unassigned.ipynb` is the concrete review artifact.

The builder rejects the failed armed notebook as a local preparation input. No old assignment is copied. Re-running the actual wrapper refuses with the inner PREPARATION ONLY error before Path.mkdir/open/write_text or subprocess.Popen can occur.

`http_error_capture.py` preserves bounded status/body and safe response headers from error.response, including HTTP400 responses whose requests.Response boolean value is false. It never serializes the request, auth headers, or cookies. Body evidence is at most65536 input bytes, UTF-8 with explicit query/credential redaction, plus original full buffered-body length/SHA256 and truncation flag. A missing response is marked UNAVAILABLE. This does not recover the lost original response.

`LAUNCH-INTEGRATION.patch` is a review-only wiring delta, not an executable launcher and not an instruction to apply against the frozen failed launcher. A future separately reviewed successor would build a newly armed inner notebook under its NEW assignment, wrap it, record/push the outer notebook hash, and capture HTTP error evidence. The original failed launcher is untouched.

## Qualification

Run `/home/pranav/.local/share/uv/tools/kaggle/bin/python verify.py` in this directory. Six checks prove full notebook byte equality, every15 embedded source hashes and aggregate source/config identities, actual unassigned wrapper refusal before extraction, corrupted digest rejection, rejection of old armed input, actual loopback HTTP400 body/status retention, credential exclusion, truncation/redaction, and missing-response behavior. Loopback is local CPU only; zero Kaggle/network calls outside loopback.

The local size assertion is <1000000 bytes to discriminate the proposed transport correction against the historical-size hypothesis. It is NOT a statement of Kaggle's current server limit. LZMA availability and current server acceptance remain untested remotely. The frozen image is Python3.12; local verification uses installed Python3.13 standard-library LZMA/XZ output, with no nonstdlib runtime dependency.

## Before any future launch

This proposal grants no launch authority. After local correction review, a separately identified successor needs its own frozen metadata/run identity, copied scientific card, launcher gate binding the entire transport+error-capture packet and unchanged scientific payload, fresh quota/session check, and explicit serialized assignment. The failed kernel directory/old assignment must not be input. Proposed future platform cap remains5400s; supervisor5100s cooperative with the accepted timing limitation and no hard5100claim. Current request is CPU review only; no cap reservation, extra cases, retry, rate/control/adoption claim, competition submission, or production promotion.
