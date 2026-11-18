"""CPU all-corpus rendering/token bound audit; never loads model weights.

Image token geometry is derived from downloaded processor/model configuration.
This does not qualify the installed runtime's actual pixel implementation.
"""
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
import jinja2
import tokenizers
from tokenizers import Tokenizer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"analysis/mtp-depth-20261006"
sys.path.insert(0,str(OLD))
import payload
import replay
import sim


def template(directory):
    env=jinja2.Environment(trim_blocks=True,lstrip_blocks=True,extensions=["jinja2.ext.loopcontrols"])
    def error(message):raise jinja2.TemplateError(message)
    env.globals["raise_exception"]=error
    env.filters["tojson"]=lambda value,indent=None,ensure_ascii=False:json.dumps(value,indent=indent,ensure_ascii=ensure_ascii)
    return env.from_string((directory/"chat_template.jinja").read_text())


def messages_for_template(messages):
    result=json.loads(json.dumps(messages))
    for message in result:
        for call in message.get("tool_calls",[]):
            function=call.get("function",{})
            if isinstance(function.get("arguments"),str):
                function["arguments"]=json.loads(function["arguments"])
    return result


def main():
    signal.alarm(1800)
    began=time.monotonic()
    data=OLD/"kaggle/dataset";games=sim.build(json.loads((data/"workload.json").read_text()))
    bodies=replay.Bodies(data,games)
    templates={}; toks={}; geometry={}
    for label in ("control","candidate"):
        directory=HERE/"processor-assets"/label
        templates[label]=template(directory);toks[label]=Tokenizer.from_file(str(directory/"tokenizer.json"))
        cfg=json.loads((directory/"preprocessor_config.json").read_text())
        factor=cfg["patch_size"]*cfg["merge_size"]
        # Every corpus image is exactly 640x640; both configured resize bounds
        # include that area, and both spatial factors divide 640 exactly.
        assert 640%factor==0 and cfg["size"]["shortest_edge"]<=640*640<=cfg["size"]["longest_edge"]
        geometry[label]={"source_hw":[640,640],"config_derived_resized_hw":[640,640],
                         "config_derived_grid_thw":[1,640//cfg["patch_size"],640//cfg["patch_size"]],
                         "image_tokens":(640//factor)**2,"spatial_factor":factor,
                         "actual_runtime_processor_verified":False}
    summary={"tokenizers_version":tokenizers.__version__,"jinja2_version":jinja2.__version__,
             "weights_loaded":False,"gpu_used":False,"geometry":geometry,"requests":0,
             "render_mismatches":0,"token_count_differences":0,"max_abs_token_count_difference":0,
             "max_joint_tokens":dict(control=0,candidate=0),"over_context":dict(control=0,candidate=0),
             "context_limit":139264,"runtime_processor_qualification":"PENDING: configs and rendering only"}
    out=(HERE/"processor-fit-records.jsonl").open("w")
    try:
        for game, value in sorted(games.items()):
            for record in value["reqs"]:
                index=record["line_index"];body=payload.build(bodies.get(game,index),"flashnext",record["completion_tokens"])
                messages=messages_for_template(body["messages"])
                rendered={label:tpl.render(messages=messages,tools=body.get("tools"),add_generation_prompt=True,
                                           **body.get("chat_template_kwargs",{})) for label,tpl in templates.items()}
                nimages=rendered["control"].count("<|image_pad|>")
                row={"game":game,"index":index,"images":nimages,"completion_cap":record["completion_tokens"],
                     "recorded_prompt_tokens":record["prompt_tokens"],"render_sha256":{},"prompt_tokens":{}}
                for label,text in rendered.items():
                    ids=toks[label].encode(text,add_special_tokens=False).ids
                    pad=toks[label].token_to_id("<|image_pad|>")
                    assert ids.count(pad)==nimages
                    count=len(ids)+nimages*(geometry[label]["image_tokens"]-1)
                    row["render_sha256"][label]=hashlib.sha256(text.encode()).hexdigest()
                    row["prompt_tokens"][label]=count
                    joint=count+record["completion_tokens"]
                    summary["max_joint_tokens"][label]=max(summary["max_joint_tokens"][label],joint)
                    summary["over_context"][label]+=joint>139264
                difference=abs(row["prompt_tokens"]["control"]-row["prompt_tokens"]["candidate"])
                summary["requests"]+=1;summary["token_count_differences"]+=difference>0
                summary["max_abs_token_count_difference"]=max(summary["max_abs_token_count_difference"],difference)
                summary["render_mismatches"]+=rendered["control"]!=rendered["candidate"]
                out.write(json.dumps(row)+"\n")
            out.flush();print(game,summary["requests"],round(time.monotonic()-began,1),flush=True)
        summary["complete"]=True
    finally:
        out.close();summary["elapsed_s"]=time.monotonic()-began
        (HERE/"processor-fit.json").write_text(json.dumps(summary,indent=2)+"\n")
        signal.alarm(0)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
