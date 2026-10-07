"""Offline prompt-fidelity check: do replayed requests carry production-sized prompts?

/tmp/tokcheck-venv/bin/python tokcheck.py [run_dir] [n_per_game]

Renders logged requests through the model's own chat template (control keys stripped,
exactly what replay.py will send), tokenizes them, and compares text tokens against the
recorded usage (prompt_tokens - image_tokens). Image pads count once in the rendering
and are excluded on both sides.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from payload import wire_messages  # noqa: E402

TOK_DIR = Path.home() / ".cache/flashnext-tok"
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).parent / "../m2-winners-20261001/runs/out-control")
N_PER_GAME = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def for_template(messages):
    """SGLang json-decodes tool_call arguments before rendering; do the same."""
    out = []
    for m in wire_messages(messages):
        if m.get("tool_calls"):
            m = dict(m)
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def template():
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii)
    return env.from_string((TOK_DIR / "chat_template.jinja").read_text())


def main():
    tpl = template()
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    pad_id = tok.token_to_id("<|image_pad|>")
    rng = random.Random(0)
    ratios, rows = [], []
    for f in sorted(RUN.glob("*_p0_requests.jsonl")):
        pairs, pending = [], None
        for line in f.open():
            r = json.loads(line)
            if r["event"] == "request":
                pending = r
            else:
                pairs.append((pending, r))
        for req, resp in rng.sample(pairs, min(N_PER_GAME, len(pairs))):
            text = tpl.render(messages=for_template(req["messages"]), tools=req.get("tools"),
                              add_generation_prompt=True, enable_thinking=True,
                              preserve_thinking=True)
            ids = tok.encode(text, add_special_tokens=False).ids
            ours = len(ids) - ids.count(pad_id)
            u = resp["usage"]
            img = (u.get("prompt_tokens_details") or {}).get("image_tokens") or 0
            theirs = u["prompt_tokens"] - img
            ratios.append(ours / theirs)
            rows.append((f.name[:4], req["analysis_step"], ours, theirs, img))
    q = st.quantiles(ratios, n=20)
    print(f"{len(ratios)} requests: ours/recorded text tokens median {st.median(ratios):.4f} "
          f"p5 {q[0]:.4f} p95 {q[-1]:.4f} min {min(ratios):.4f} max {max(ratios):.4f}")
    worst = sorted(rows, key=lambda r: abs(r[2] / r[3] - 1), reverse=True)[:8]
    for w in worst:
        print("worst", w, f"{w[2] / w[3]:.3f}")


if __name__ == "__main__":
    main()
