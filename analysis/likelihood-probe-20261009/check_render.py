"""Fidelity: local template+tokenizer count (images expanded to 400 pads) vs the server's logged prompt_tokens."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_items import TEMPLATE, TOK, RUNS, clean
f = sorted(glob.glob(str(RUNS["dfz-control"] / "*_p0_requests.jsonl")))[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
L = [json.loads(l) for l in open(f)]
for i in range(0, len(L) - 1):
    a, b = L[i], L[i + 1]
    if a["event"] != "request" or b["event"] != "response" or i % 40: continue
    msgs = []
    for m in a["messages"]:
        m2 = clean([m])[0]
        if isinstance(m.get("content"), list): m2["content"] = m["content"]   # keep images as placeholders
        msgs.append(m2)
    s = TEMPLATE.render(messages=msgs, tools=a["tools"], add_generation_prompt=True, **a.get("chat_template_kwargs", {}))
    n = len(TOK.encode(s, add_special_tokens=False).ids); img = s.count("<|image_pad|>")
    print(f.split("/")[-1][:4], i, "local", n + img * 399, "server", b["usage"]["prompt_tokens"], "img_tokens", b["usage"]["prompt_tokens_details"].get("image_tokens"), img)
