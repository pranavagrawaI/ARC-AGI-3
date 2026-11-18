"""Create the private Kaggle dataset from the completed gb202 blob session (token only, no local bytes).

The dataset page is a LIL-1.0 Covered Page (§1.9(c), "wherever it is hosted"), so its description
opens with the Attribution Notice from the NOTICE of the Original Location (§3.2), and there is no
subtitle in front of it. The bytes are the unmodified LFS object, so the safetensors metadata is kept
(§3.6). The dataset stays private (§2.4 permission, PERMISSION.md).

  ~/.local/share/uv/tools/kaggle/bin/python -B create_dataset.py gb202
"""
import json
import sys

import session
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiDatasetNewFile

OWNER = "pranavagrawai"
SLUG = "flare-gb202-package-20261020"
TITLE = "flare gb202 package 20261020"
NOTICE = (
    "This dataset is based on Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD by Local Inference Lab, Inc., a "
    "non-profit organization, available at "
    "[https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD]"
    "(https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD). "
    "Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0."
)
DESCRIPTION = NOTICE + """

Private, unmodified copy of the `gb202` revision of
[local-inference-lab/Qwen3.8-Flash-Next-FLARE](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-FLARE)
(`qwen3.8-flash-next-tp1-c16-mtp5.safetensors`, 107,011,338,717 bytes, sha256
c3db5e1ac63b05efa67d6525826a8a792c0ce92b810e91ddd5ec0f42c73a175c), kept with written permission from
Local Inference Lab, Inc. for the owner's own notebooks and ARC Prize 2026 submissions. Not for sharing.

Upstream Materials: Qwen/Qwen3.8-Flash-Next (quantized), licensed under qwen-community-1.0,
https://huggingface.co/Qwen/Qwen3.8-Flash-Next.
"""


def main():
    rec = session.load(sys.argv[1])
    state = session.status(rec)
    if state[0] != "COMPLETE":
        sys.exit(f"blob not complete: {state}")
    req = ApiCreateDatasetRequest()
    req.title, req.slug, req.owner_slug = TITLE, SLUG, OWNER
    req.license_name = "other"
    req.description = DESCRIPTION
    req.is_private = True
    f = ApiDatasetNewFile()
    f.token = rec["token"]
    req.files = [f]
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        resp = kaggle.datasets.dataset_api_client.create_dataset(req)
    out = {"ref": resp.ref, "url": resp.url, "status": resp.status, "error": resp.error}
    print(json.dumps(out, indent=1))
    if resp.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
