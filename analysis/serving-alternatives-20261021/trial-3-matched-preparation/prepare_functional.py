"""Select and hash six diagnostic prefix identities locally. No launch freeze."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
import processor_fit as fit


def main():
    here=Path(__file__).resolve().parent
    records=[json.loads(line) for line in (here/"processor-fit-records.jsonl").read_text().splitlines()]
    priority=[max(records,key=lambda r:r["prompt_tokens"]["control"]),
              max(records,key=lambda r:r["images"]),
              max(records,key=lambda r:r["prompt_tokens"]["control"]+r["completion_cap"])]
    priority+=sorted(records,key=lambda r:(-r["images"],-r["prompt_tokens"]["control"],r["game"],r["index"]))
    selected=[];seen=set()
    for row in priority:
        key=(row["game"],row["index"])
        if key not in seen:seen.add(key);selected.append(row)
        if len(selected)==6:break
    data=fit.OLD/"kaggle/dataset"
    bodies=fit.replay.Bodies(data,{r["game"] for r in selected})
    transform={"append_user_text":"Diagnostic only: call the python tool with exactly the code print(1). Do not solve the game or add explanation.",
               "max_tokens":256,"temperature":0.,"top_p":1.,
               "tool_choice":{"type":"function","function":{"name":"python"}},
               "chat_template_kwargs":{"enable_thinking":False,"preserve_thinking":True}}
    result={"purpose":"functional interface/trace/long-input check only; never a rate or quality sample",
            "transformation":transform,"concurrency":2,"phase_cap_s":300,"cases":[]}
    templates={};toks={}
    for label in ("control","candidate"):
        directory=here/"processor-assets"/label
        templates[label]=fit.template(directory);toks[label]=Tokenizer.from_file(str(directory/"tokenizer.json"))
    for row in selected:
        body=fit.payload.build(bodies.get(row["game"],row["index"]),"flashnext",256)
        body["messages"].append({"role":"user","content":transform["append_user_text"]})
        body.update({k:v for k,v in transform.items() if k!="append_user_text"})
        counts={}
        for label,tpl in templates.items():
            text=tpl.render(messages=fit.messages_for_template(body["messages"]),tools=body["tools"],
                            add_generation_prompt=True,**body["chat_template_kwargs"])
            ids=toks[label].encode(text,add_special_tokens=False).ids
            counts[label]=len(ids)+row["images"]*399
            assert counts[label]+256<=139264
        canonical=json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
        result["cases"].append(dict(game=row["game"],index=row["index"],images=row["images"],
                                    request_sha256=hashlib.sha256(canonical).hexdigest(),prompt_tokens=counts))
    (here/"functional-cases.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
