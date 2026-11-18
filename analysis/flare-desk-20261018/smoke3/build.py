"""Build the stage-3 smoke kernel: smoke.py as one code cell under a LIL-1.0 Attribution Notice cell.

Same image, machine and inputs as 2b, except the six part outputs are replaced by the single-file
private dataset flare-gb202-package-20261020.

  python3 -B build.py   # writes kernel/{flare-smoke-3-20261020.ipynb, kernel-metadata.json}
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "flare-smoke-3-20261020"
NOTICE = (
    "This project is based on Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD by Local Inference Lab, Inc., a "
    "non-profit organization, available at "
    "[https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD]"
    "(https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD). "
    "Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0."
)


def main():
    code = (HERE / "smoke.py").read_text()
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
                kernel_sources=[])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(d, meta["dataset_sources"])


if __name__ == "__main__":
    main()
