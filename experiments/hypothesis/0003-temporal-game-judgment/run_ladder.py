from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from game_judgment import ARMS, ContractError, canonical_bytes, digest, parse_decision, score, visible_case  # noqa:E402

def main():
 p=argparse.ArgumentParser(); p.add_argument("--responses",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 batch=json.loads((ROOT/"fixtures/batch.json").read_text()); labels={x["case_id"]:x for x in json.loads((ROOT/"scorer/labels.json").read_text())["labels"]}
 supplied={(x["case_id"],x["arm"],x.get("repeat",0)):x for x in map(json.loads,a.responses.read_text().splitlines())}; a.output.mkdir(parents=True,exist_ok=False); results=[]
 for case in batch["cases"]:
  for arm in ARMS:
   request=visible_case(case,arm); key=(case["case_id"],arm,0); row={"case_id":key[0],"arm":arm,"request_sha256":digest(request)}
   try:
    raw=supplied[key]["raw_response"]; decision=parse_decision(raw); row.update({"raw_response":raw,"parser":{"tool":decision.tool,"arguments":decision.arguments},"score":score(case,labels[key[0]],decision)})
   except (KeyError,ContractError) as e: row.update({"parser_error":str(e),"score":{"contract":False,"pass":False}})
   results.append(row)
 (a.output/"requests.jsonl").write_bytes(b"".join(canonical_bytes({"case_id":c["case_id"],"arm":arm,"request":visible_case(c,arm)})+b"\n" for c in batch["cases"] for arm in ARMS))
 (a.output/"results.json").write_bytes(canonical_bytes({"schema":"0003-ladder-result-v1","environment_actions":0,"results":results})+b"\n")
if __name__=="__main__": main()
