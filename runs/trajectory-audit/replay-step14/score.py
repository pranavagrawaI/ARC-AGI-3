"""Mechanical pre-screen for the two rubric criteria; final scoring is by reading (SCORES.md).

C1 fresh-data: some TOOL RESULT in the trial contains the BR centre rows printed from code, i.e. the
   six lines WWgggg/WWgggg/WWRRWW/WWRRWW/WWgggg/WWgggg in sequence (any surrounding chars allowed), and
   a later THINKING/TEXT/CODE segment states the BR centre as "W g g / W R W / W g g" (or the decode).
C2 reference-check: reasoning/code compares the mapping against TL, TR and BL before acting
   (heuristic: a THINKING/TEXT/CODE segment mentions all three of TL/Top-left, TR/Top-right,
   BL/Bottom-left together with 'center'/'centre' and 'outer' or 'map').
Also reports: decode line found, executed actions, simulator completion.
"""
import json, re, sys, glob, os
def secs(t):
    parts = re.split(r"(?m)^\[([A-Z :a-z]+)\]$", t); return [(parts[j], parts[j + 1]) for j in range(1, len(parts) - 1, 2)]
BR_LINES = ["WWgggg", "WWgggg", "WWRRWW", "WWRRWW", "WWgggg", "WWgggg"]
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*-*"))):
    tp = os.path.join(d, "transcript.txt"); rp = os.path.join(d, "result.json")
    if not os.path.exists(tp): continue
    t = open(tp, encoding="utf-8").read(); r = json.load(open(rp)) if os.path.exists(rp) else {}
    s = secs(t)
    printed = False
    for l, c in s:
        if l.startswith("TOOL RESULT"):
            lines = [x.strip() for x in c.splitlines()]
            for i in range(len(lines) - 5):
                if all(BR_LINES[k] in lines[i + k] for k in range(6)) and not any("ggggWW" in lines[i + k] for k in range(6)):
                    # make sure it's the BR (cols 44-49) not BL: BL has ggggWW in rows 48-49
                    printed = True
    reasoning = " ".join(c for l, c in s if l in ("THINKING", "ASSISTANT") or l.startswith("TOOL CALL"))
    decode_ok = bool(re.search(r"R b b\s*/\s*R \? R\s*/\s*R b b", reasoning))
    decode_bad = bool(re.search(r"R b b\s*/\s*R \? R\s*/\s*b b R", reasoning)) and "BR" in reasoning
    ref = bool(re.search(r"(?is)(TL|top-left).{0,400}(TR|top-right).{0,400}(BL|bottom-left).{0,300}(cent(er|re)).{0,300}(outer|map)", reasoning))
    wgg = "W g g / W R W / W g g" in reasoning
    print(f"{os.path.basename(d):14} C1_printed_BR={int(printed)} states_Wgg/WRW/Wgg={int(wgg)} decode_correct={int(decode_ok)} decode_BLcopy={int(decode_bad)} C2_ref_check~={int(ref)} "
          f"actions={len(r.get('actions',[]))} completed={int(bool(r.get('level_completed')))} s={r.get('seconds')}")
