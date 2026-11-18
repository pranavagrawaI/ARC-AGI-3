"""Compare the screen's first system prompt and tool schema to archived A0 requests."""
import json
import base64
import io
from PIL import Image
import screen


def main():
    root = screen.HERE / "evidence/qualified-dry-all"
    rows = []
    observations = []
    for key in screen.starts():
        old = screen.ORACLE / "sessions/a0/output" / f"{key}-A0-r1_p0_requests.jsonl"
        with old.open() as f:
            baseline = json.loads(next(f))
        new = root / key / "receipts/000000.json.gz"
        request = screen.load(new)["request"]
        a, b = baseline["messages"][0]["content"], request["messages"][0]["content"]
        rows.append({"key": key, "system_prompt_equal": a == b, "python_tools_equal": baseline["tools"] == request["tools"],
                     "baseline_system_sha256": screen.sha(a.encode()), "screen_system_sha256": screen.sha(b.encode()),
                     "first_user_message_equal": baseline["messages"][-1] == request["messages"][-1],
                     "baseline_request": str(old), "screen_request": str(new)})
        old_user, new_user = baseline["messages"][-1], request["messages"][-1]
        def texts(message):
            return [part["text"] for part in message["content"] if part["type"] == "text"]
        def pixels(message):
            return [Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
                    for part in message["content"] if part["type"] == "image_url"]
        a_images, b_images = pixels(old_user), pixels(new_user)
        observations.append({"key": key, "user_text_equal": texts(old_user) == texts(new_user),
                             "image_count_equal": len(a_images) == len(b_images),
                             "image_pixels_equal": len(a_images) == len(b_images) and all(a.size == b.size and a.tobytes() == b.tobytes() for a, b in zip(a_images, b_images))})
    result = {"selected": len(rows), "system_prompts_equal": sum(r["system_prompt_equal"] for r in rows),
              "tool_schemas_equal": sum(r["python_tools_equal"] for r in rows), "rows": rows,
              "scope": "first model request of each of the 40 starts; no claim of byte-identical serving, clocks or scheduler"}
    screen.atomic(screen.HERE / "evidence/harness-fidelity-all.json", result)
    observation_result = {"selected": len(observations), "user_text_equal": sum(r["user_text_equal"] for r in observations),
                          "image_pixels_equal": sum(r["image_pixels_equal"] for r in observations), "rows": observations,
                          "note": "PNG encodings differ; compare decoded RGB pixels instead of base64 bytes"}
    screen.atomic(screen.HERE / "evidence/observation-fidelity-all.json", observation_result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
