# Kaggle q6 control preflight

A private, single-run package around the qualified q6 candidate. `preflight.ipynb` is byte-identical to the frozen control notebook. This package does not launch during preparation or verification.

From the repository root:

```bash
python3 deployment/kaggle/q6-control-preflight/prepare.py --check
/home/pranav/.local/share/uv/tools/kaggle/bin/python deployment/kaggle/q6-control-preflight/verify_request.py
kaggle quota --format json
```

`verify_request.py` calls the installed Kaggle 2.2.4 request builder with its transport replaced. It verifies the private kernel, RTX Pro 6000 shape, 14400-second server timeout and control/action settings without contacting the server. CLI and metadata alone do not establish GPU availability.

After launch authorization, execute the exact argv in `LAUNCH.json` once from the repository root. Recheck package hashes and quota immediately beforehand. Preserve the push response; an ambiguous network response is not permission to retry. Reconcile kernel status first.

```bash
kaggle kernels push --path deployment/kaggle/q6-control-preflight --timeout 14400 --accelerator NvidiaRtxPro6000
kaggle kernels status pranavagrawai/q6-control-preflight-20260905
kaggle kernels output pranavagrawai/q6-control-preflight-20260905 --path analysis/deployment-probes/20260905-q6-control-preflight/raw
python3 analysis/local-probes/20260905T-stall-authority-q6/score_q6.py analysis/deployment-probes/20260905-q6-control-preflight/raw --output analysis/deployment-probes/20260905-q6-control-preflight/SCORE.json
```

Download after terminal status; retain all logs even on failure. No complete session artifacts means invalid, not zero stall incidence. Check model/runtime logs and host evidence before interpreting the scorer. The single control run cannot establish q6 treatment efficacy and is excluded from the later arm.

Run question, caps and interpretation: `analysis/deployment-probes/20260905-q6-control-preflight/EXPERIMENT-CARD.md`.
