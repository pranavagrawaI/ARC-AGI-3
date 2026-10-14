"""Hash-bound overlay. Control restores exact wheel bytes; candidate changes one routing seam."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

WHEEL_SHA = "d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225"
MEMBER = "sglang/srt/models/qwen2_moe.py"
SOURCE_SHA = "441a5e992c7299a0a6be1a3268706759062f6a15bf9a0c8350e24def051c4c2f"
OLD = """        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )"""
NEW = """        from sglang.srt.layers.moe.reference_routing import reference_routing_for
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
            custom_routing_function=reference_routing_for(config, is_nextn, layer_id),
        )"""


def candidate_source(original):
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise ValueError("routing source identity mismatch")
    text = original.decode()
    if text.count(OLD) != 1:
        raise ValueError("routing seam is not unique")
    return text.replace(OLD, NEW).encode()


def install(wheel, site, arm, receipt):
    wheel, site = Path(wheel), Path(site)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise ValueError("wheel identity mismatch")
    with zipfile.ZipFile(wheel) as archive:
        original = archive.read(MEMBER)
    changed = candidate_source(original)
    target = site / MEMBER
    if target.read_bytes() not in (original, changed):
        raise ValueError("installed runtime has unrelated modifications")
    if arm not in ("control", "candidate"):
        raise ValueError("unknown arm")
    target.write_bytes(original if arm == "control" else changed)
    helper = Path(__file__).with_name("routing.py").read_bytes()
    (site / "sglang/srt/layers/moe/reference_routing.py").write_bytes(helper)
    result = {"arm": arm, "wheel_sha256": WHEEL_SHA, "original_source_sha256": SOURCE_SHA,
              "installed_source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(helper).hexdigest(), "control_exact_wheel_bytes": arm == "control"}
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
    print("REFERENCE_ROUTING_INSTALLED", json.dumps(result), flush=True)


if __name__ == "__main__":
    install(*sys.argv[1:])
