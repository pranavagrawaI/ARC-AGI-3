"""Parse P's final prompt snapshots into chat sessions for the serving-benchmark replay.

  python3 bench/build_sessions.py [out-p1]      # -> bench/sessions.json

Each `out-p1/prompts/<game>_p0.log` holds the rendered MODEL INPUT of the last request of that game
(tool_agent._render_prompt_log_message). This inverts that renderer and checks the inversion by re-rendering
every message and comparing with the original text byte for byte.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / (sys.argv[1] if len(sys.argv) > 1 else "out-p1")
HEADER = re.compile(r"^\[(SYSTEM|USER|ASSISTANT|TOOL RESULT: [^\]]+)\]$")
CALL = re.compile(r"^\[ASSISTANT TOOL CALL: (\w+)\]$")


def tool_description(text: str) -> str:
    start = text.index("\n[AVAILABLE TOOLS]\n- python: ") + len("\n[AVAILABLE TOOLS]\n- python: ")
    end = text.index("\n\n[MODEL INPUT]\n", start)
    return text[start:end]


def model_input(text: str) -> str:
    start = text.index("\n[MODEL INPUT]\n") + len("\n[MODEL INPUT]\n")
    end = text.index("\n\n[TURN TRANSCRIPT SO FAR]\n", start)
    return text[start:end]


def split_messages(body: str) -> list[tuple[str, str]]:
    lines = body.split("\n")
    starts = [i for i, line in enumerate(lines) if HEADER.match(line) and (i == 0 or lines[i - 1] == "")]
    out = []
    for n, i in enumerate(starts):
        j = starts[n + 1] - 1 if n + 1 < len(starts) else len(lines)   # drop the "\n\n" joiner's blank line
        out.append((lines[i][1:-1], "\n".join(lines[i + 1:j])))
    return out


def parse_call(block: list[str]) -> dict:
    # block: ["id: X", "<tool_call>", "<function=python>", "<parameter=code>", ...code..., "</parameter>", "</function>", "</tool_call>"]
    assert block[0].startswith("id: ") and block[1] == "<tool_call>" and block[-1] == "</tool_call>", block[:3]
    name = re.fullmatch(r"<function=(\w+)>", block[2])[1]
    args, k = {}, 3
    while block[k] != "</function>":
        pname = re.fullmatch(r"<parameter=(\w+)>", block[k])[1]
        end = block.index("</parameter>", k)
        args[pname] = "\n".join(block[k + 1:end])
        k = end + 1
    return {"id": block[0][4:], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def parse_assistant(text: str) -> dict:
    lines = text.split("\n")
    calls = [i for i, line in enumerate(lines) if CALL.match(line)]
    body_end = calls[0] if calls else len(lines)
    body = lines[:body_end]
    msg = {"role": "assistant", "content": ""}
    if "[REASONING]" in body:
        r = body.index("[REASONING]")
        msg["content"], msg["reasoning_content"] = "\n".join(body[:r]), "\n".join(body[r + 1:])
    else:
        msg["content"] = "\n".join(body)
    if calls:
        bounds = calls + [len(lines)]
        msg["tool_calls"] = [parse_call(lines[a + 1:b]) for a, b in zip(bounds, bounds[1:])]
    return msg


def to_message(role: str, text: str) -> dict:
    if role == "ASSISTANT":
        return parse_assistant(text)
    if role.startswith("TOOL RESULT: "):
        return {"role": "tool", "tool_call_id": role[len("TOOL RESULT: "):], "content": text}
    return {"role": role.lower(), "content": text}


def render(msg: dict) -> str:
    """Mirror of tool_agent._render_prompt_log_message for the fields this parser produces."""
    role = msg["role"].upper()
    blocks = [f"[TOOL RESULT: {msg['tool_call_id']}]" if role == "TOOL" else f"[{role}]"]
    if msg.get("content"):
        blocks.append(msg["content"])
    if msg.get("reasoning_content"):
        blocks += ["[REASONING]", msg["reasoning_content"]]
    for call in msg.get("tool_calls", []):
        blocks += [f"[ASSISTANT TOOL CALL: {call['function']['name']}]", f"id: {call['id']}", "<tool_call>",
                   f"<function={call['function']['name']}>"]
        for k, v in json.loads(call["function"]["arguments"]).items():
            blocks += [f"<parameter={k}>"] + ([v] if v else []) + ["</parameter>"]
        blocks += ["</function>", "</tool_call>"]
    return "\n".join(blocks)


def main() -> None:
    sessions, bad, descriptions = {}, [], set()
    for path in sorted((RUN / "prompts").glob("*_p0.log")):
        text = path.read_text()
        body = model_input(text)
        descriptions.add(tool_description(text))
        msgs = [to_message(r, t) for r, t in split_messages(body)]
        if "\n\n".join(render(m) for m in msgs) != body:
            bad.append(path.name)
        assert msgs[0]["role"] == "system" and msgs[-1]["role"] in {"user", "tool"}, (path.name, msgs[-1]["role"])
        game = path.name.split("_p0")[0]
        sessions[game] = {
            "messages": msgs,
            "user_pool": [m["content"] for m in msgs if m["role"] == "user"],
            "tool_pool": [m["content"] for m in msgs if m["role"] == "tool"],
            "chars": len(body),
        }
    systems = {s["messages"][0]["content"] for s in sessions.values()}
    report = {"games": len(sessions), "roundtrip_mismatch": bad, "distinct_system_prompts": len(systems),
              "chars_mean": sum(s["chars"] for s in sessions.values()) / len(sessions),
              "min_tool_pool": min(len(s["tool_pool"]) for s in sessions.values()),
              "min_user_pool": min(len(s["user_pool"]) for s in sessions.values())}
    print(json.dumps(report, indent=1))
    report["distinct_tool_descriptions"] = len(descriptions)
    assert not bad and len(systems) == 1 and len(descriptions) == 1 and report["min_user_pool"] >= 1
    (HERE / "sessions.json").write_text(json.dumps({"source": str(RUN), "report": report,
                                                    "tool_description": descriptions.pop(), "sessions": sessions}))


if __name__ == "__main__":
    main()
