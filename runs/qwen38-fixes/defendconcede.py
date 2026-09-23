"""Do questions entrench? Split answered questions by whether the actor's answer defends or concedes
its belief, and compare the lock-break rate (new target within 10 actions) after each.
usage: defendconcede.py <arm dir> [<arm dir> ...]"""
import json, glob, os, re, sys, random
WINDOW = 10
CONCEDE = re.compile(r"\b(fair|good (catch|point|question)|not proven|unproven|unverified|i should (verify|test|check)|"
                     r"haven'?t (tested|checked|verified|tried)|may be|might be|could be|it doesn'?t have to|"
                     r"not necessarily|no evidence|untested|worth testing|i'?ll test)\b", re.I)
def target(a):
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", a)
    return f"M{int(m.group(1))//8},{int(m.group(2))//8}" if m else a
rows = []
for d in sys.argv[1:]:
    for tr in glob.glob(f"{d}/transcripts/*critic.jsonl"):
        stem = os.path.basename(tr).replace("_critic.jsonl", "")
        ledger = None
        for l in open(tr): ledger = json.loads(l)["ledger"]
        acts = [e for e in (json.loads(l) for l in open(f"{d}/artifacts/{stem}_events.jsonl")) if e["type"] == "action"]
        for e in ledger or []:
            if not e["reply"]: continue
            p = e["action_num"] - 1; after = acts[p:p + WINDOW]
            if not after: continue
            lv = int(after[0].get("level") or 1)
            before = {target(a["action_display"]) for a in acts[:p] if int(a.get("level") or 1) == lv}
            rows.append(("concede" if CONCEDE.search(e["reply"]) else "defend",
                         any(target(a["action_display"]) not in before for a in after), e["question"], e["reply"]))
for k in ("defend", "concede"):
    x = [r for r in rows if r[0] == k]
    print(f"{k}: n={len(x)}, break rate {100*sum(r[1] for r in x)/max(len(x),1):.0f}%")
random.seed(1)
for k in ("defend", "concede"):
    print("--", k)
    for r in random.sample([r for r in rows if r[0] == k], 3):
        print("  Q:", r[2][:110], "\n  A:", r[3][:160])
