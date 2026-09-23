"""Run the questioner over frozen packets against a local vLLM (Kaggle). Writes results.json.
Independent mode: every packet with an empty ledger, efforts none/low. Sequence mode: sp80 and
tn36 packets in order with the ledger carried (effort low), to exercise repeat-avoidance."""
import json, os, sys, time, concurrent.futures as cf
import requests
BASE = os.environ.get("GRADE_BASE", "/kaggle/working")
sys.path.insert(0, BASE + "/grade")
from questioner import QUESTIONER_SYSTEM_PROMPT, QUESTION_FORMS, QuestionLedger, parse_question, render_questioner_input, validate_question

URL = os.environ.get("GRADE_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B-FP8"
packets = json.load(open(BASE + "/grade/packets.json"))

def call(messages, effort):
    kw = {"enable_thinking": effort != "none"}
    if effort != "none": kw["reasoning_effort"] = effort
    payload = {"model": MODEL, "messages": messages, "max_tokens": 3000, "temperature": 0.6, "top_p": 0.95,
               "top_k": 20, "chat_template_kwargs": kw}
    r = requests.post(URL, json=payload, timeout=600); r.raise_for_status(); b = r.json()
    m = b["choices"][0]["message"]
    return str(m.get("content") or ""), str(m.get("reasoning") or m.get("reasoning_content") or ""), b.get("usage", {})

def ask(p, effort, ledger, form=None):
    prompt = render_questioner_input(level=p["level"], action_num=p["action_num"], turns_since_level=p["turn"],
                                     recent_words=p["recent_words"], transitions=p["transitions"], ledger=ledger, form=form)
    messages = [{"role": "system", "content": QUESTIONER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    t0 = time.time(); attempts = []; q = ""
    for _ in range(2):
        content, reasoning, usage = call(messages, effort)
        cand = parse_question(content); rej = validate_question(cand, ledger)
        attempts.append({"content": content[:400], "reasoning_tail": reasoning[-600:], "candidate": cand, "rejection": rej,
                         "completion_tokens": usage.get("completion_tokens")})
        if not rej: q = cand; break
        messages += [{"role": "assistant", "content": content}, {"role": "user", "content": f"Rejected: {rej}. One question only."}]
    return {"game": p["game"], "turn": p["turn"], "effort": effort, "form": form, "question": q, "seconds": round(time.time() - t0, 1),
            "attempts": attempts, "prompt_chars": len(prompt)}

results = []
jobs = [(p, e, QUESTION_FORMS[i % len(QUESTION_FORMS)]) for e in ("none", "low") for i, p in enumerate(packets)]
with cf.ThreadPoolExecutor(24) as ex:
    for r in ex.map(lambda j: ask(j[0], j[1], QuestionLedger(), j[2]), jobs):
        results.append({**r, "mode": "independent"}); print(r["game"], r["turn"], r["effort"], "|", r["question"], flush=True)
for g in ("sp80", "tn36"):
    ledger = QuestionLedger()
    for p in [p for p in packets if p["game"] == g]:
        r = ask(p, "low", ledger)
        if r["question"]: ledger.add(p["level"], p["action_num"], r["question"])
        results.append({**r, "mode": "sequence"}); print("SEQ", g, p["turn"], "|", r["question"], flush=True)
json.dump(results, open(BASE + "/results.json", "w"), indent=1)
print("done", len(results))
