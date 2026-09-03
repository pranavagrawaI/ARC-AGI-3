"""Local checks for the vision-readback probe.

The probe's whole value is that its numbers are trustworthy, so these tests attack the
places where it could quietly lie: a grid text that is not the one V15 actually sent, a
PNG whose pixels do not carry the palette, ground truth derived from the answer rather
than the frame, and a scorer that credits a missing tool call.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

CANDIDATE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CANDIDATE))

import vision_readback_probe as probe  # noqa: E402

REFERENCE_WORKSPACE = Path(
    "/home/pranav/projects/arc-agi-3-reference/tycho/tycho/workspace/workspace.py"
)


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    doc = json.loads((CANDIDATE / "frames.json").read_text())
    assert doc["schema"] == "tycho-nooa-vision-readback-frames-v1"
    return doc["frames"]


@pytest.fixture(scope="module")
def grid(frames) -> list[list[int]]:
    return frames[0]["grid"]


# --- the pinned asset -------------------------------------------------------------


def test_frames_are_three_distinct_64x64_ls20_levels(frames):
    assert len(frames) == 3
    assert {f["level"] for f in frames} == {0, 3, 5}
    for frame in frames:
        assert len(frame["grid"]) == 64
        assert all(len(row) == 64 for row in frame["grid"])
        assert all(0 <= v < 16 for row in frame["grid"] for v in row)


# --- the text arm must be the incumbent, not a lookalike --------------------------


@pytest.mark.skipif(
    not REFERENCE_WORKSPACE.is_file(), reason="reference Tycho checkout not present"
)
def test_grid_text_matches_reference_spaced_grid8(grid):
    """Byte equality with tycho/workspace/workspace.py, loaded without importing Tycho."""
    source = REFERENCE_WORKSPACE.read_text()
    start = source.index("def grid_text(")
    end = source.index("def grid_hex(")
    namespace: dict = {"GRID_FORMAT": "spaced_grid8"}
    exec(compile(source[start:end], str(REFERENCE_WORKSPACE), "exec"), namespace)
    assert probe.grid_text(grid) == namespace["grid_text"](grid)


def test_grid_text_labels_every_row_and_separates_column_blocks(grid):
    lines = probe.grid_text(grid).splitlines()
    assert lines[0].startswith("x:")
    assert len(lines) == 65
    assert lines[1].startswith("y00: ")
    assert lines[64].startswith("y63: ")
    assert lines[1].count("|") == 7


# --- the image arm must actually carry the palette --------------------------------


@pytest.mark.parametrize("scale", probe.RENDER_SCALES)
def test_png_round_trips_every_cell_at_every_scale(grid, scale):
    """Decode the PNG back and confirm each cell's centre pixel is its palette colour."""
    pillow = pytest.importorskip("PIL.Image")
    image = pillow.open(io.BytesIO(probe.grid_png(grid, scale))).convert("RGB")
    assert image.size == (64 * scale, 64 * scale)
    half = scale // 2
    for r in (0, 31, 61, 63):
        for c in (0, 8, 32, 63):
            expected = probe.PALETTE_16[grid[r][c]]
            assert image.getpixel((c * scale + half, r * scale + half)) == expected


def test_scale_32_render_is_the_size_the_lossless_claim_requires(grid):
    """4,194,304 px is the threshold tycho/agent/vision.py conditions its claim on."""
    pillow = pytest.importorskip("PIL.Image")
    image = pillow.open(io.BytesIO(probe.grid_png(grid, 32)))
    assert image.size == (2048, 2048)
    assert image.size[0] * image.size[1] == 4_194_304


# --- ground truth is derived from the frame, and covers the hard cases ------------


def test_cell_sample_is_deterministic_and_covers_hud_rows(grid):
    cells = probe.sample_cells(grid)
    assert cells == probe.sample_cells(grid)
    assert len(cells) == probe.CELL_SAMPLE_SIZE
    assert len({tuple(c) for c in cells}) == len(cells)
    assert any(r == 61 for r, _ in cells), "the depleting bar rows must be sampled"
    assert [0, 0] in cells and [63, 63] in cells


def test_cell_sample_is_not_winnable_by_guessing_one_colour(frames):
    """Keeps the headline accuracy meaningful.

    The dominant colour covers most of an ls20 frame, so a clustered sample would let a
    model that answers "4" twelve times score most of it without seeing anything.
    """
    for frame in frames:
        truth = probe.truth_for(frame["grid"], "cells")
        most_common = Counter(truth).most_common(1)[0][1]
        assert most_common / len(truth) <= 0.5, frame["record_index"]
        assert len(set(truth)) >= 4, frame["record_index"]


def test_cell_sample_is_spread_across_the_frame(frames):
    for frame in frames:
        rows = {r for r, _ in probe.sample_cells(frame["grid"])}
        assert max(rows) - min(rows) >= 32, frame["record_index"]


def test_truth_matches_the_frame_array(grid):
    values = probe.truth_for(grid, "cells")
    assert values == [grid[r][c] for r, c in probe.sample_cells(grid)]

    palette = probe.truth_for(grid, "palette")
    assert palette == sorted({v for row in grid for v in row})

    r0, c0, r1, c1 = probe.truth_for(grid, "bbox")
    color = probe.bbox_color(grid)
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color]
    assert cells, "the bbox colour must actually occur"
    assert (r0, c0) == (min(r for r, _ in cells), min(c for _, c in cells))
    assert (r1, c1) == (max(r for r, _ in cells), max(c for _, c in cells))


def test_bbox_colour_is_a_real_object_not_a_stray_pixel(grid):
    color = probe.bbox_color(grid)
    count = sum(1 for row in grid for v in row if v == color)
    assert count >= 4


# --- requests carry the contract the condition declares ---------------------------


def test_conditions_cover_text_image_and_both():
    names = {c.name for c in probe.CONDITIONS}
    assert names == {
        "text_nothink",
        "image06_nothink",
        "image16_nothink",
        "image32_nothink",
        "both32_nothink",
        "text_think",
    }
    incumbent = next(c for c in probe.CONDITIONS if c.name == "text_think")
    assert incumbent.max_tokens == 16000 and not incumbent.disable_thinking
    assert incumbent.frame_limit == 1, "the expensive arm must not run on every frame"


@pytest.mark.parametrize("condition", probe.CONDITIONS, ids=lambda c: c.name)
def test_request_shape_follows_its_condition(condition, grid):
    request = probe.build_request("m", condition, grid, "cells")
    parts = request["messages"][1]["content"]
    kinds = [p["type"] for p in parts]
    assert kinds.count("image_url") == (1 if condition.render_scale else 0)
    # one instruction block, plus the grid text when the condition inlines it
    assert kinds.count("text") == (2 if condition.show_text else 1)
    assert request["tool_choice"] == "required"
    assert request["max_tokens"] == condition.max_tokens
    assert [t["function"]["name"] for t in request["tools"]] == ["report_cells"]
    if condition.disable_thinking:
        assert request["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    else:
        assert "extra_body" not in request


def test_image_conditions_send_a_png_data_url(grid):
    condition = next(c for c in probe.CONDITIONS if c.name == "image32_nothink")
    request = probe.build_request("m", condition, grid, "bbox")
    url = request["messages"][1]["content"][-1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_bbox_instruction_names_the_colour_it_scores(grid):
    text = probe.task_instruction("bbox", grid)
    assert str(probe.bbox_color(grid)) in text


# --- the scorer credits nothing it should not -------------------------------------


def test_missing_or_malformed_tool_call_scores_zero(grid):
    truth = probe.truth_for(grid, "cells")
    assert probe.score("cells", truth, None) == {
        "answered": False,
        "exact": False,
        "accuracy": 0.0,
    }
    short = probe.score("cells", truth, {"values": [1, 2]})
    assert short["accuracy"] == 0.0 and short["malformed"] is True


def test_cell_scoring_is_positional(grid):
    truth = probe.truth_for(grid, "cells")
    assert probe.score("cells", truth, {"values": truth})["accuracy"] == 1.0
    assert probe.score("cells", truth, {"values": list(reversed(truth))})["exact"] is (
        truth == list(reversed(truth))
    )
    partial = list(truth)
    partial[0] = (partial[0] + 1) % 16
    assert probe.score("cells", truth, {"values": partial})["accuracy"] == round(
        (len(truth) - 1) / len(truth), 4
    )


def test_palette_scoring_uses_set_equality_and_jaccard():
    assert probe.score("palette", [1, 2, 3], {"colors": [3, 2, 1]})["exact"] is True
    partial = probe.score("palette", [1, 2, 3], {"colors": [1, 2]})
    assert partial["exact"] is False and partial["accuracy"] == round(2 / 3, 4)


def test_bbox_scoring_reports_exactness_and_iou():
    truth = [10, 10, 12, 12]
    exact = probe.score("bbox", truth, dict(zip(
        ("min_row", "min_col", "max_row", "max_col"), truth)))
    assert exact["exact"] is True and exact["accuracy"] == 1.0
    disjoint = probe.score("bbox", truth, {
        "min_row": 40, "min_col": 40, "max_row": 41, "max_col": 41})
    assert disjoint["accuracy"] == 0.0


# --- end to end against a fake server ---------------------------------------------


class _PerfectHandler(BaseHTTPRequestHandler):
    """Answers every task correctly, so a passing run exercises the whole path."""

    grid: list[list[int]] = []

    def log_message(self, *args):  # keep the test output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = body["tools"][0]["function"]["name"]
        grid = type(self).grid
        if name == "report_cells":
            args = {"values": probe.truth_for(grid, "cells")}
        elif name == "report_palette":
            colors = probe.truth_for(grid, "palette")
            # the token-measurement probe asks for a trivial palette answer
            args = {"colors": colors if body["max_tokens"] > 16 else [0]}
        else:
            keys = ("min_row", "min_col", "max_row", "max_col")
            args = dict(zip(keys, probe.truth_for(grid, "bbox")))
        has_image = any(
            p["type"] == "image_url" for p in body["messages"][-1]["content"]
        )
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {"function": {"name": name, "arguments": json.dumps(args)}}
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100 + (4096 if has_image else 0),
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def fake_server(grid):
    _PerfectHandler.grid = grid
    server = HTTPServer(("127.0.0.1", 0), _PerfectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.fixture
def one_frame_asset(frames, tmp_path) -> Path:
    """The fake responder knows one grid, so pin the probe to that same grid.

    Feeding it all three frames would score frame 0's answers against frames 3 and 5 and
    measure the fixture rather than the probe.
    """
    path = tmp_path / "frames.json"
    path.write_text(json.dumps({
        "schema": "tycho-nooa-vision-readback-frames-v1",
        "game_id": "ls20-9607627b",
        "frames": frames[:1],
    }))
    return path


def test_end_to_end_scores_a_perfect_responder(
    fake_server, one_frame_asset, tmp_path, monkeypatch
):
    monkeypatch.setattr(probe, "CONDITIONS", probe.CONDITIONS[:2])
    output = tmp_path / "result.json"
    argv = [
        "probe",
        "--base-url", fake_server,
        "--model", "test-model",
        "--frames", str(one_frame_asset),
        "--output", str(output),
        "--request-timeout-s", "30",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert probe.main() == 0

    result = json.loads(output.read_text())
    assert result["status"] == "passed"
    assert result["schema"] == probe.PROBE_SCHEMA
    for name, stats in result["summary"].items():
        assert stats["mean_accuracy"] == 1.0, name
        assert stats["exact"] == stats["requests"], name
    assert len(result["records"]) == 2 * len(probe.TASK_NAMES)


def test_end_to_end_penalises_a_responder_that_answers_the_wrong_frame(
    fake_server, tmp_path, monkeypatch
):
    """Guards the guard: scoring must fall when the answers stop matching the frame."""
    monkeypatch.setattr(probe, "CONDITIONS", probe.CONDITIONS[:1])
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "probe", "--base-url", fake_server, "--model", "m",
        "--frames", str(CANDIDATE / "frames.json"), "--output", str(output),
    ])
    assert probe.main() == 0
    summary = json.loads(output.read_text())["summary"]["text_nothink"]
    assert summary["requests"] == 3 * len(probe.TASK_NAMES)
    assert summary["mean_accuracy"] < 1.0
    assert summary["exact"] < summary["requests"]


def test_end_to_end_reports_the_measured_visual_token_cost(
    fake_server, one_frame_asset, tmp_path, monkeypatch
):
    """The gating number must come from a real prompt-token difference, not a constant."""
    monkeypatch.setattr(probe, "CONDITIONS", probe.CONDITIONS[:1])
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "probe", "--base-url", fake_server, "--model", "m",
        "--frames", str(one_frame_asset), "--output", str(output),
    ])
    assert probe.main() == 0

    measurement = json.loads(output.read_text())["visual_token_measurement"]
    assert measurement["text_only_prompt_tokens"] == 100
    by_scale = {m["render_scale"]: m for m in measurement["scales"]}
    assert set(by_scale) == set(probe.RENDER_SCALES)
    assert by_scale[32]["image_pixels"] == 4_194_304
    assert by_scale[32]["expected_visual_tokens_if_lossless"] == 4096
    assert by_scale[32]["observed_visual_tokens"] == 4096
    assert by_scale[32]["one_token_per_cell"] is True


def test_a_downscaling_server_is_reported_as_such(grid, tmp_path, monkeypatch):
    """If SGLang shrinks the render, the probe must say so rather than pass quietly."""
    calls = {"n": 0}

    def fake_post(url, payload, timeout):
        calls["n"] += 1
        has_image = any(
            p["type"] == "image_url" for p in payload["messages"][-1]["content"]
        )
        return {
            "choices": [{"message": {"tool_calls": [
                {"function": {"name": "report_palette", "arguments": '{"colors":[0]}'}}
            ]}}],
            "usage": {"prompt_tokens": 100 + (256 if has_image else 0)},
        }

    monkeypatch.setattr(probe, "post_json", fake_post)
    measurement = probe.measure_visual_tokens("http://x/v1", "m", grid, 5)
    by_scale = {m["render_scale"]: m for m in measurement["scales"]}
    assert by_scale[32]["observed_visual_tokens"] == 256
    assert by_scale[32]["one_token_per_cell"] is False
    assert calls["n"] == 1 + len(probe.RENDER_SCALES)


def test_transport_failure_is_recorded_not_raised(grid, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(probe, "post_json", boom)
    condition = probe.CONDITIONS[0]
    frame = {"record_index": 0, "level": 0}
    record = probe.run_one("http://x/v1", "m", condition, frame, "cells", grid, 1)
    assert record["score"]["answered"] is False
    assert "connection refused" in record["error"]
