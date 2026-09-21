# 20260921-v12-sb26-critic-off (Kaggle)

Same card as `EXPERIMENT-CARD.md`. One private kernel, no submission, no retry.

- Kernel: `pranavagrawai/tycho-nooa-v12-sb26-critic-off`
- Machine: NvidiaRtxPro6000, internet off, session 10800 s
- Command: `kaggle kernels push --path deployment/kaggle/submission-runtime-20260920-v12 --timeout 10800 --accelerator NvidiaRtxPro6000`
- Games: sb26; 2 sequential passes; 1800 s / 200 actions each
- Factor: epistemic critic off, contingent off; v12 eviction keep=4
