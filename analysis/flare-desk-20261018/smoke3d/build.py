"""Build the stage-3d smoke kernel: smoke.py as one code cell under a LIL-1.0 Attribution Notice cell.

flare_ra.so (../readahead/flare_ra.c, built and tested there) is embedded zlib+base64 with its sha256.
Inputs: the 2b bundle and replay data, the single-file dataset (route A), and the six 10-18 part
outputs (route B, staged to local disk). The smoke benchmarks both routes and loads from the faster one.

  python3 -B build.py   # writes kernel/{flare-smoke-3b-20261020.ipynb, kernel-metadata.json}
"""
import base64
import hashlib
import json
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "flare-smoke-3d-20261020"
PARTS = [f"pranavagrawai/flare-gb202-part{i}{'b' if i == 5 else ''}-20261018" for i in range(6)]
NOTICE = (
    "This project is based on Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD by Local Inference Lab, Inc., a "
    "non-profit organization, available at "
    "[https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD]"
    "(https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD). "
    "Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0."
)


def main():
    so = (HERE.parent / "readahead" / "flare_ra.so").read_bytes()
    code = (HERE / "smoke.py").read_text()
    assert code.count("__RA_SO_B64__") == 1 and code.count("__RA_SO_SHA__") == 1
    code = (code.replace("__RA_SO_B64__", base64.b64encode(zlib.compress(so, 9)).decode())
                .replace("__RA_SO_SHA__", hashlib.sha256(so).hexdigest()))
    compile(code, "smoke.py", "exec")
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [NOTICE]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    d = HERE / "kernel"
    d.mkdir(exist_ok=True)
    (d / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((HERE.parent / "smoke2b" / "kernel" / "kernel-metadata.json").read_text())
    meta.update(id=f"pranavagrawai/{SLUG}", title=SLUG.replace("-", " "), code_file=f"{SLUG}.ipynb",
                dataset_sources=["pranavagrawai/flare-probe-bundle-20261018",
                                 "pranavagrawai/mtp-depth-replay-data-20261006",
                                 "pranavagrawai/flare-gb202-package-20261020"],
                kernel_sources=PARTS)
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(d, meta["dataset_sources"])


if __name__ == "__main__":
    main()
