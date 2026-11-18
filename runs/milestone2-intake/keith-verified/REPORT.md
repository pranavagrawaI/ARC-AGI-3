# Milestone 2 notebook intake

Status: **prepared_unexecuted**

Candidate: `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`
Duck: `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`
Private destination: `pranavagrawai/m2-duck-qwen3-8-flash-n-20260925-165415-995708`

## Requirements

- dataset_sources: `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`, `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`
- model_sources: `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`
- kernel_sources: none declared
- competition_sources: `arc-prize-2026-arc-agi-3`

Runtime settings:

```json
{
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "machine_shape": "NvidiaRtxPro6000",
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
  "docker_image_pinning_type": null
}
```

Python imports (module names, not a pip requirements file): arc_agi, datetime, inference.tools.eval, json, os, pandas, pathlib, pickle, subprocess, sys, sysconfig, taaf.game_api, time, urllib.request, vllm_server_watchdog

### Install commands

Code cell 2, line 3:

```text
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)
```

### Input paths

Code cell 2, line 14:

```text
/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels
```

Code cell 3, line 15:

```text
/kaggle/input/datasets/
```

Code cell 3, line 20:

```text
/kaggle/input/datasets
```

Code cell 7, line 96:

```text
/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels
```

### Urls

Code cell 7, line 89:

```text
http://gateway:8001/
```

### Secrets

None detected.

### Environment variables

Code cell 1, line 13:

```text
KAGGLE_IS_COMPETITION_RERUN
```

Code cell 1, line 17:

```text
MPLBACKEND
```

Code cell 1, line 19:

```text
TAAF_RUN_AS_SUBMISSION
```

Code cell 1, line 21:

```text
TAAF_MINIMAL_DIAGNOSTICS
```

Code cell 1, line 43:

```text
ONLY_RESET_LEVELS
```

Code cell 1, line 48:

```text
LIBRARY_PATH
```

Code cell 1, line 49:

```text
LIBRARY_PATH
```

Code cell 4, line 44:

```text
PYTHONPATH
```

Code cell 7, line 9:

```text
ARC_BASE_URL
```

Code cell 7, line 91:

```text
ARC_BASE_URL
```

### External code

Code cell 4, line 26:

```text
# Make the bundled repos importable here (sys.path) and in child processes (.pth).
```

Code cell 4, line 29:

```text
sys.path.insert(0, str(entry))
```

Code cell 4, line 45:

```text
if entry not in sys.path:
```

Code cell 4, line 46:

```text
sys.path.insert(0, entry)
```

Code cell 5, line 3:

```text
target = pickle.load(file)
```

Code cell 5, line 9:

```text
bm = pickle.load(file)
```

Code cell 7, line 124:

```text
if str(BUNDLE_DIR) not in sys.path:
```

Code cell 7, line 125:

```text
sys.path.insert(0, str(BUNDLE_DIR))
```

### Parse gaps

None detected.

## Differences from Duck

Executable cell diff: `code.diff`; configuration diff: `settings.diff`.
Notebook outputs and execution counts are excluded from the code diff.

```json
{
  "settings": {},
  "attachments": {
    "dataset_sources": {
      "added": [
        "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
        "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
      ],
      "removed": [
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        "jeroencottaar/taaf-kaggle-source-share"
      ]
    },
    "model_sources": {
      "added": [
        "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
      ],
      "removed": []
    },
    "kernel_sources": {
      "added": [],
      "removed": []
    },
    "competition_sources": {
      "added": [],
      "removed": []
    }
  },
  "imports_added": [
    "inference.tools.eval",
    "vllm_server_watchdog"
  ],
  "imports_removed": [
    "IPython.display",
    "html"
  ],
  "code_identical": false
}
```

## Private copy changes

```json
{
  "docker_image_pinning_type": {
    "original": null,
    "private": "original"
  },
  "id": {
    "original": "keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp",
    "private": "pranavagrawai/m2-duck-qwen3-8-flash-n-20260925-165415-995708"
  },
  "id_no": {
    "original": 132392822,
    "private": null
  },
  "is_private": {
    "original": false,
    "private": true
  },
  "title": {
    "original": "Duck Qwen3.8 Flash Next NVFP4 MTP",
    "private": "m2 duck qwen3 8 flash n 20260925 165415 995708"
  }
}
```

- Upgrade notebook format to 4.5 for cell IDs
- Set unique ID for notebook cell 0
- Set unique ID for notebook cell 1
- Set unique ID for notebook cell 2
- Set unique ID for notebook cell 3
- Set unique ID for notebook cell 4
- Set unique ID for notebook cell 5
- Set unique ID for notebook cell 6
- Set unique ID for notebook cell 7
- Set unique ID for notebook cell 8
- Set unique ID for notebook cell 9
- Set unique ID for notebook cell 10
- Set unique ID for notebook cell 11
- Set unique ID for notebook cell 12
- Set unique ID for notebook cell 13
- Set unique ID for notebook cell 14
- Set unique ID for notebook cell 15
- Set unique ID for notebook cell 16
- Set unique ID for notebook cell 17

## Unresolved checks

- Static notebook inspection only. Attached solver code, wheels, weights, pickles and transitive dependencies were not downloaded or executed; this is not a full solver diff.
- Attachment accessibility, competition consent, secrets, available GPU/quota, and successful execution remain unverified.
- Dataset references returned by Kaggle pull may be unversioned. Preserved bytes/hashes pin the notebooks, not all input contents.

## Commit command (not executed)

```bash
kaggle kernels push --path /home/pranav/projects/arc-agi-3/runs/milestone2-intake/keith-verified/kernel --accelerator NvidiaRtxPro6000
```

A push creates and runs a Kaggle version. Review the report, input access and quota before authorizing it.
