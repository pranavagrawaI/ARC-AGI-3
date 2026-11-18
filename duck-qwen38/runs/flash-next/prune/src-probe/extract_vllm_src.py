"""CPU-only: pull Keith's vLLM model/loader source out of the pinned runtime image layers (no GPU)."""
import tarfile, re, io, json
from pathlib import Path

roots = [p for p in Path("/kaggle/input").rglob("runtime-manifest.json")]
print("manifest", roots)
root = roots[0].parent
want = re.compile(r"(vllm/models/qwen3_8_flash_next/|vllm/model_executor/layers/fused_moe/(layer|config)\.py|"
                  r"vllm/config/speculative\.py|vllm/model_executor/model_loader/(default_loader|weight_utils|utils)\.py|"
                  r"vllm/v1/spec_decode/.*\.py$|vllm/_version\.py|vllm/version\.py|vllm/model_executor/models/(qwen3_next|qwen3_5|utils)\.py|vllm/plugins/__init__\.py|vllm/model_executor/models/qwen3_8_flash_next.*\.py)")
out = tarfile.open("/kaggle/working/vllm-src.tar.gz", "w:gz")
listing = {}
for blob in sorted(root.glob("layer-*.blob")):
    n = 0
    try:
        with tarfile.open(blob, "r|gz") as tf:
            for m in tf:
                if re.search(r"(site|dist)-packages/vllm/models/", m.name) and m.name.count("/") < 12:
                    listing.setdefault("model_dirs", set()).add(re.split(r"(?:site|dist)-packages/", m.name)[1].split("/")[2])
                if m.isfile() and re.search(r"(site|dist)-packages/", m.name) and want.search(m.name):
                    f = tf.extractfile(m)
                    data = f.read()
                    info = tarfile.TarInfo(re.split(r"(?:site|dist)-packages/", m.name, 1)[1]); info.size = len(data)
                    out.addfile(info, io.BytesIO(data)); n += 1
    except Exception as exc:
        print(blob.name, "error", exc)
    print(blob.name, "extracted", n, flush=True)
out.close()
print(json.dumps({k: sorted(v) for k, v in listing.items()}))
