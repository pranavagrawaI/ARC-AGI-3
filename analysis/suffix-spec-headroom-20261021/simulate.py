"""Offline headroom for training-free suffix/prompt-lookup speculation on logged ARC requests.

For every distinct assistant message in the logged request histories, replay its tokens left to right.
At each step, propose the continuation of the most recent earlier occurrence of the last n tokens
(n = 8..3) in context+generated-so-far, accept the longest matching prefix, and yield accepted+1 tokens.

MTP3 already yields 2.67 accepted tokens per verify step (max 4), so suffix proposals only add value
where they accept >= 4 tokens. Two bounds on a hybrid (suffix when it proposes >= 4, else MTP):
  optimistic:   MTP averages 2.67 on the remaining tokens
  conservative: MTP was perfect (4/step) on the copy spans, so the rest cost T/2.67 - T_copy/4 steps

Usage: python simulate.py TOKENIZER_JSON REQUESTS_JSONL...
"""

import hashlib
import json
import sys

from tokenizers import Tokenizer

MTP_RATE = 2.67
MTP_MAX = 4
K = 32
NS = (8, 7, 6, 5, 4, 3)


def message_text(m):
    parts = []
    for key in ("reasoning_content", "content"):
        v = m.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts += [p.get("text", "") for p in v if isinstance(p, dict)]
    for call in m.get("tool_calls") or []:
        parts.append(call.get("function", {}).get("arguments", ""))
    return "\n".join(parts)


def simulate(ctx, gen):
    seq = list(ctx)
    index = {n: {} for n in NS}

    def insert(end):  # n-grams ending at `end`; the current suffix (end == len(seq)) is never indexed
        for n in NS:
            if end >= n:
                index[n][tuple(seq[end - n:end])] = end

    for end in range(1, len(seq)):
        insert(end)
    t = 0
    steps = copy_steps = copy_tokens = 0
    while t < len(gen):
        accepted = 0
        for n in NS:
            if len(seq) < n:
                continue
            p = index[n].get(tuple(seq[-n:]))
            if p is None:
                continue
            proposal = seq[p:p + K]
            while accepted < len(proposal) and t + accepted < len(gen) and proposal[accepted] == gen[t + accepted]:
                accepted += 1
            break
        take = min(accepted + 1, len(gen) - t)
        steps += 1
        if accepted >= MTP_MAX:
            copy_steps += 1
            copy_tokens += take
        for tok in gen[t:t + take]:
            seq.append(tok)
            insert(len(seq) - 1)
        t += take
    return steps, copy_steps, copy_tokens


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    seen = set()
    tot = {"T": 0, "steps": 0, "copy_steps": 0, "copy_tokens": 0, "messages": 0}
    per_file = {}
    for path in sys.argv[2:]:
        f = {"T": 0, "copy_tokens": 0, "copy_steps": 0}
        for line in open(path):
            msgs = json.loads(line).get("messages", [])
            for j, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                text = message_text(m)
                h = hashlib.sha1(text.encode()).hexdigest()
                if h in seen or len(text) < 50:
                    continue
                seen.add(h)
                ctx_text = "\n".join(message_text(x) if x.get("role") == "assistant" else
                                     (x["content"] if isinstance(x.get("content"), str) else
                                      "\n".join(p.get("text", "") for p in x.get("content") or [] if isinstance(p, dict)))
                                     for x in msgs[:j])
                ctx = tok.encode(ctx_text).ids
                gen = tok.encode(text).ids
                s, cs, ct = simulate(ctx, gen)
                for d in (tot, f):
                    d["T"] += len(gen)
                    d["copy_steps"] += cs
                    d["copy_tokens"] += ct
                tot["steps"] += s
                tot["messages"] += 1
        per_file[path] = f
    T, Tc, Sc = tot["T"], tot["copy_tokens"], tot["copy_steps"]
    base = T / MTP_RATE
    out = {
        "messages": tot["messages"],
        "generated_tokens": T,
        "suffix_only_tokens_per_step": round(T / tot["steps"], 3),
        "copy_fraction": round(Tc / T, 4),
        "tokens_per_copy_step": round(Tc / max(Sc, 1), 2),
        "hybrid_speedup_optimistic": round(base / (Sc + (T - Tc) / MTP_RATE), 4),
        "hybrid_speedup_conservative": round(base / (Sc + base - Tc / MTP_MAX), 4),
        "per_file_copy_fraction": {p.split("analysis/")[-1]: round(v["copy_tokens"] / max(v["T"], 1), 4)
                                   for p, v in per_file.items()},
        "per_file_tokens": {p.split("analysis/")[-1]: v["T"] for p, v in per_file.items()},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
