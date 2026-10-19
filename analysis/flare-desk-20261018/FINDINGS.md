# FLARE serving runtime: desk check (2026-10-18, no GPU, binary not executed)

## Why it matters
dfranzen serving is step-time-bound, and levels-time elasticity is about 0.35. A runtime that decodes
1.3-1.5x faster at our concurrency is worth roughly +10-17% levels, which is the silver-sized gap
(draw-variance-20261018). Every other open serving lever is spent (c14, MTP depth, REAP, Swift, Marlin).

## What FLARE is
- `local-inference-lab/Qwen3.8-Flash-Next-FLARE`, created 2026-10-16, alpha community preview. It is
  a prepared runtime package for Flare, a native Rust server: NVFP4 routed experts + MXFP8
  attention/shared experts (base `...-NVFP4-MXFP8-CSF-QAD`, i.e. quantization-aware distilled),
  MTP speculative decoding up to 5 draft tokens, compiled CUDA kernels and captured graphs.
- **There is a `gb202` revision for the RTX PRO 6000 96 GB, which is Kaggle's GPU.** It is prepared
  for one GPU and up to 16 concurrent requests. The 26.8 GiB PLE table is held in RAM or read from disk.
- No throughput is published for Flare (the lab's rtx6kpro wiki has none). For reference, the same
  lab measured its vLLM build at C16 944.5 tok/s, context 0, NVFP4 + B12X + MTP3
  (rtx6kpro/models/qwen38-flash-next/validation/tp1-engine-comparison.md). Our incumbent is
  815 tok/s at 14 streams with real long contexts. The two are not comparable.

## Portability evidence (from the image, pulled via the anonymous registry API; binary NOT executed)
- `ghcr.io/local-inference-lab/flare:latest`, built 2026-10-18 00:29Z. The image is 38 MB: Debian
  trixie, libgcc only, plus a single 8.4 MB stripped Rust binary (`/usr/local/bin/flare`,
  sha256 7d2ad472...a9d5).
- Dynamic deps: libc, libm, libgcc_s. It needs **GLIBC_2.39**. At runtime it loads libcuda.so.1 and
  libnvidia-ml.so.1 (driver only, no CUDA toolkit). If Kaggle's image has an older glibc, ship the
  trixie loader + libc and run it through `ld-linux-x86-64.so.2 --library-path`.
- API strings: /v1/chat/completions, /v1/responses, /v1/messages; tool_calls, tool_choice,
  reasoning_content, reasoning_effort; prefix cache (RAM/disk).
- Flags include --max-concurrency, --mtp-draft-tokens, --no-mtp-adaptive-verification,
  --kv-cache-limit, --embedding-table-storage, --prefix-cache-disk, --verify-weights and
  --trust-store. **Offline unattended start:** "unattended first use requires
  --dangerously-approve-key FINGERPRINT --dangerously-accept-publisher NAMESPACE". The trust store is
  local (~/.config/flare/trust.toml). The only network host string is huggingface.co (downloads).

## Blockers
1. **License (LIL-1.0) forbids Reupload.** No copy may be made available to "any other person" by any
   means, including model hubs and cloud storage (2.1). Permitted: "keeping copies on devices and in
   storage that only You control" (2.2(b)). A **private** Kaggle dataset plausibly falls under (b); a
   public one does not. Written permission from an LIL officer can authorize a Reupload (2.4).
   Pranav's call. The license of the Flare binary itself is not stated in the image.
2. **Size.** The gb202 package is one 107.0 GB safetensors file, and the laptop has 98 GB free.
   RTX sessions must run with internet off, so even the smoke needs the package as an attached
   private dataset. Community Flash-Next mirrors of 119 GB exist on Kaggle, so that size is feasible.
3. Unknowns that need a GPU: decode tok/s on our replayed request mix at 10/14/16 streams vs SGLang
   W4A16; tool-call format vs the dfranzen harness parser; NVFP4/MXFP8-QAD quality vs Intel
   W4A16-AutoRound (Swift's INT4 attention cost 54% of levels, so quality is not free).

## Decisive first experiment (once blockers 1-2 are cleared)
One GPU smoke, about 1 h: start Flare from the private dataset, then replay real B1 request logs
(`*_requests.jsonl`) at concurrency 10/14/16 and measure aggregate decode tok/s and tool-call
parse rate. Compare with the incumbent SGLang W4A16 on the same replay. Go to a 60-min screen only
if decode >= 1.3x and tool calls parse.

## Addendum 2026-10-18 (afternoon): Kaggle-side evidence

- **Portability probe** (`probe/`, kernel flare-portability-probe-20261018, ~3 min GPU). Kaggle runs
  glibc 2.35, so the native binary fails. Through the shipped trixie loader it runs (`flare 0.1.0`) and
  selects gb202 on its own. Driver 580.178.04 (CUDA 13.0) exports every symbol Flare imports. A 32 GiB
  cuMemHostRegister succeeds at memlock 8 MB. **io_uring_setup is EPERM (seccomp)**, so disk PLE and
  the disk prefix cache can't work and RAM PLE is required. /kaggle/input reads ~79 MB/s per stream.
- **Transfer** (`transfer/`). The laptop uploads at ~3.8 MB/s, which would take ~8 h. Six internet-on
  CPU kernels range-GET 16.6 GiB each into /kaggle/working in 2-4 min (batch CPU cap 5, so part 5 was
  re-id'd 5b). A CPU verifier streamed all six as kernel_sources: sha256 = LFS c3db5e1a…, bytes and
  per-part hashes OK.
- **Smoke 2a** (kernel flare-smoke-2a-20261018) ended ERROR after ~53 min with an empty log and no
  outputs, so the session was killed outright. Probe 2 (RAM) found memory.max 175 GiB, no swap,
  /dev/shm 86.5 GiB (too small), memfd works, but Flare resolves paths so `/proc/self/fd/N` fails.
  Probe 3 (disk) held fallocate 108 GB in /tmp for 90 s and survived, so the disk cap isn't the
  killer. The working hypothesis is memory: page cache from the reassembly plus whatever Flare pins.
- **Smoke 2b** (`smoke2b/`): the same card, plus page-cache dropping during reassembly and a memory
  watchdog that kills Flare above 160 GiB and logs every 30 s, so the notebook survives to report.
- **Smoke 2b** died the same way (~17 min, empty log). **Probe 4** (reassembly only, no Flare) also
  died (~15 min, empty log), so Flare isn't the killer: writing 107 GB of real data to local disk kills
  the container, while a 108 GB fallocate does not. **Probe 5** (read-only, six parts in parallel,
  hashed and discarded) survived: all 107 GB in 212 s = 504 MB/s aggregate, cgroup memory ≤1.2 GiB,
  every hash OK.
- **Consequence:** Flare must read its package in place from /kaggle/input, and it accepts one file, so
  the package must exist on Kaggle as a single 107 GB dataset/model file. Routes: (a) laptop streaming
  HF → Kaggle blob upload, ~8 h at 3.8 MB/s, laptop awake; (b) an internet-on CPU kernel streaming
  HF → Kaggle blob upload, which needs a Kaggle API credential as a Kaggle Secret (Pranav adds it in the
  UI); (c) Kaggle's "remote URL" dataset import from the UI, if it accepts 107 GB. All of them fall under
  the same LIL-1.0 §2.2(b) private-storage reading.
