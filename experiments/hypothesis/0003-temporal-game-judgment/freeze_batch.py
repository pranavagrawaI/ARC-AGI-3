from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
EXPECTED = "434bd2f66b5e77b6aaa3bc1ed9cb0d5e28b0070ccaf93c5ebb077eacec5bf5b4"
sys.path.insert(0, str(ROOT))
from game_judgment import canonical_bytes  # noqa: E402

PALETTE = [(0,0,0),(0,116,217),(255,65,54),(46,204,64),(255,220,0),(170,170,170),(240,18,190),(255,133,27),(127,219,255),(135,12,37)]
CONTRACT = {"tools": {
    "EXECUTE": ["plan_id", "first_action", "predicted_public_fields"],
    "PROBE": ["discriminator_id", "action", "predicted_outcomes"],
    "ABSTAIN": ["reason_code", "missing_evidence"]}, "exactly_one_call": True}

def write(path: Path, value): path.write_bytes(canonical_bytes(value) + b"\n")

def main() -> int:
    if sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED: raise RuntimeError("source digest mismatch")
    rows = [json.loads(line)["data"] for line in SOURCE.read_text().splitlines()]
    fixtures, scorer, images = ROOT/"fixtures", ROOT/"scorer", ROOT/"fixtures"/"images"
    images.mkdir(parents=True, exist_ok=True); scorer.mkdir(parents=True, exist_ok=True)
    specs = [("execution-control",13,"execution"),("planning-control",12,"planning"),("game-judgment",6,"judgment")]
    cases=[]; labels=[]
    for cid, step, burden in specs:
        image_refs=[]
        for s in range(step-3, step+1):
            grid=rows[s]["frame"][-1]; im=Image.new("RGB",(len(grid[0])*5,len(grid)*5))
            px=im.load()
            for y,line in enumerate(grid):
                for x,c in enumerate(line):
                    for yy in range(y*5,y*5+5):
                        for xx in range(x*5,x*5+5): px[xx,yy]=PALETTE[int(c)%len(PALETTE)]
            p=images/f"{cid}-step-{s}.png"; im.save(p); image_refs.append({"step":s,"path":str(p.relative_to(ROOT)),"sha256":sha256(p.read_bytes()).hexdigest(),"size":list(im.size)})
        action=lambda s: int(str(rows[s]["action_input"]["id"]).replace("ACTION",""))
        transitions=[{"before_step":s-1,"after_step":s,"action":action(s),"changed_cells":sum(a!=b for ra,rb in zip(rows[s-1]["frame"][-1],rows[s]["frame"][-1]) for a,b in zip(ra,rb))} for s in range(step-2,step+1)]
        plans=[{"plan_id":"route-historical","actions":[action(step+1)],"prediction":{"public.levels_completed":rows[step+1]["levels_completed"]},"conjecture_ids":["C-translation","O-progress"]}]
        case={"schema":"0003-decision-case-v1","case_id":cid,"burden":burden,"source":{"path":str(SOURCE.relative_to(REPO)),"sha256":EXPECTED,"frontier":step},"legal_actions":rows[step]["available_actions"],"current_public":{k:rows[step][k] for k in ("state","levels_completed","win_levels","available_actions")},"images":image_refs,"step_labels":[x["step"] for x in image_refs],"numeric_transitions":transitions,"reflection":"Movement changes one lattice position; preserve unresolved interaction alternatives.","candidate_scene":{"neutral_tracks":["track-0"],"regions":["region-a","region-b"]},"executable_conjectures":[{"id":"C-translation","relation":"action_translation","domain":"track-0","direction":"forward","parameters":{"step":5},"evidence_refs":[f"transition-{step-1}"],"predicted_output":"next_position"},{"id":"O-progress","relation":"entry_correlates_with_progress","domain":"region-b","direction":"forward","parameters":{},"evidence_refs":[],"predicted_output":"public.levels_completed+1","replay":"uncovered"}],"candidate_plans":plans,"verified_route":plans[0],"discriminators":[{"discriminator_id":"cycle-entry","action":action(step+1),"predicted_outcomes":{"advance":"latent+1","no_advance":"latent"}}],"contract":CONTRACT}
        cases.append(case)
        successor={"public.levels_completed":rows[step+1]["levels_completed"]}
        labels.append({"case_id":cid,"protected_successor":successor,"valid_decision_signatures":[["EXECUTE","route-historical",action(step+1)],["ABSTAIN","insufficient_evidence"]]})
    write(fixtures/"batch.json",{"schema":"0003-frozen-batch-v1","source_sha256":EXPECTED,"cases":cases})
    write(scorer/"labels.json",{"schema":"0003-scorer-labels-v1","labels":labels})
    print(fixtures/"batch.json"); return 0

if __name__ == "__main__": raise SystemExit(main())
