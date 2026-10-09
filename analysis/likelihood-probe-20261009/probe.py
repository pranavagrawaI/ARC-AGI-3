"""Teacher-forced likelihood probe, run on Kaggle against a live SGLang server (no sampling).

  $PYTHON probe.py ITEMS.jsonl.xz OUT.jsonl --model-dir DIR --kf-system KF.txt --url http://127.0.0.1:8001 --budget-s 14400

For each item (one logged decision point) and each variant, every candidate's code is appended to the rendered context
and scored with /generate return_logprob (prompt logprobs only). Variants, in priority order:
  R0      dfranzen context, empty think block            -> intuitive choice
  R1      dfranzen context, the model's own logged reasoning for that turn
  KF0     R0 with KF's (Duck) system prompt swapped in   -> known-worse prompt on the same model (hidden 2.96 vs 23.87)
  MIN0    R0 with a one-line system prompt               -> sensitivity check
Writes one JSON line per (item, variant) as soon as it is scored, so a timeout keeps everything done so far.
"""
import argparse, concurrent.futures as cf, json, lzma, sys, time, urllib.request
from pathlib import Path
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

OPEN = "\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n"
CLOSE = "\n</parameter>\n"
MIN_SYSTEM = "You are a coding agent solving a grid-based puzzle game."


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Renderer:
    def __init__(self, template_text, tok_path):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)
        def _raise(m): raise jinja2.TemplateError(m)
        env.globals["raise_exception"] = _raise
        self.t = env.from_string(template_text)
        self.tok = Tokenizer.from_file(str(tok_path))

    def prompt(self, messages, tools):
        return self.t.render(messages=messages, tools=tools, add_generation_prompt=True, preserve_thinking=True)

    def ids(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids


def openai_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": c["function"]["name"],
                                "arguments": json.dumps(c["function"]["arguments"])}} for i, c in enumerate(m["tool_calls"])]
        out.append(m)
    return out


def fidelity(R, items, url, model):
    for it in items[:3]:
        n = len(R.ids(R.prompt(it["messages"], it["tools"])))
        try:
            r = post(f"{url}/v1/chat/completions", dict(model=model, messages=openai_messages(it["messages"]), tools=it["tools"],
                     max_tokens=1, temperature=0, chat_template_kwargs={"preserve_thinking": True}))
            print(f"[fidelity] {it['id']} local={n} server={r['usage']['prompt_tokens']} diff={r['usage']['prompt_tokens'] - n}", flush=True)
        except Exception as e:
            print(f"[fidelity] {it['id']} local={n} server ERROR {e!r}", flush=True)


def score(url, prefix_ids, cand_ids):
    full = prefix_ids + cand_ids
    r = post(f"{url}/generate", {"input_ids": full, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                                 "return_logprob": True, "logprob_start_len": len(prefix_ids) - 1})
    lp = r["meta_info"]["input_token_logprobs"]
    tail = lp[-len(cand_ids):]
    assert [t[1] for t in tail] == cand_ids, f"alignment: got {[t[1] for t in tail][:5]} want {cand_ids[:5]}"
    return sum(t[0] for t in tail), [round(t[0], 4) for t in tail]


def variant_messages(it, v, kf_system):
    msgs = [dict(m) for m in it["messages"]]
    if v in ("KF0", "MIN0") and msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = kf_system if v == "KF0" else MIN_SYSTEM
    return msgs


def run_item(R, url, it, v, kf_system, pool):
    if v == "R1" and not it.get("reasoning"): return None
    p = R.prompt(variant_messages(it, v, kf_system), it["tools"])
    head = p + (it["reasoning"].strip() if v == "R1" else "") + OPEN
    prefix = R.ids(head)
    cands = []
    for c in it["cands"]:
        full = R.ids(head + c["code"] + CLOSE)
        assert full[:len(prefix)] == prefix, "tokenization boundary moved"
        cands.append(full[len(prefix):])
    t0 = time.time()
    first = score(url, prefix, cands[0])                           # warm the shared prefix once
    rest = list(pool.map(lambda ci: score(url, prefix, ci), cands[1:]))
    lps = [first] + rest
    return dict(id=it["id"], variant=v, prefix_tokens=len(prefix), secs=round(time.time() - t0, 2),
                lp=[round(x[0], 4) for x in lps], tok_lp=[x[1] for x in lps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--model-dir", required=True); ap.add_argument("--kf-system", required=True)
    ap.add_argument("--local-template"); ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="flashnext"); ap.add_argument("--budget-s", type=float, default=4 * 3600)
    ap.add_argument("--variants", default="R0,R1,KF0,MIN0"); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t_start = time.time()
    md = Path(a.model_dir); tmpl = md / "chat_template.jinja"
    text = tmpl.read_text() if tmpl.is_file() else json.loads((md / "tokenizer_config.json").read_text())["chat_template"]
    if a.local_template:
        print(f"[template] model-dir template {'==' if text == Path(a.local_template).read_text() else '!='} local copy", flush=True)
    R = Renderer(text, md / "tokenizer.json")
    items = [json.loads(l) for l in lzma.open(a.items, "rt")]
    kf_system = Path(a.kf_system).read_text()
    # game-major order keeps consecutive items in one context lineage (prefix cache) while covering every game early
    items.sort(key=lambda it: (it["game"], it["step"]))
    print(f"[probe] {len(items)} items, variants {a.variants}", flush=True)
    fidelity(R, items, a.url, a.model)
    done = set()
    if Path(a.out).exists():
        done = {(j["id"], j["variant"]) for j in map(json.loads, open(a.out))}
    with cf.ThreadPoolExecutor(a.workers) as pool, open(a.out, "a") as out:
        for v in a.variants.split(","):
            n_ok = n_err = 0; tv = time.time()
            for it in items:
                if time.time() - t_start > a.budget_s:
                    print(f"[probe] budget reached in {v}", flush=True); return
                if (it["id"], v) in done: continue
                try:
                    row = run_item(R, a.url, it, v, kf_system, pool)
                except Exception as e:
                    n_err += 1; print(f"[probe] {v} {it['id']} ERROR {e!r}"[:400], flush=True)
                    if n_err > 20 and n_ok == 0: raise SystemExit(f"variant {v}: 20 errors and no success")
                    continue
                if row is None: continue
                out.write(json.dumps(row) + "\n"); out.flush(); n_ok += 1
                if n_ok % 25 == 1:
                    print(f"[probe] {v} {n_ok} ok {n_err} err  last {row['id']} prefix={row['prefix_tokens']} {row['secs']}s  "
                          f"elapsed {time.time() - tv:.0f}s", flush=True)
            print(f"[probe] variant {v} finished: {n_ok} ok, {n_err} err, {time.time() - tv:.0f}s", flush=True)


if __name__ == "__main__":
    main()
