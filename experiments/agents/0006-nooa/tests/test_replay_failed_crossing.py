"""The failed-crossing replay must rebuild faithfully or refuse."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from base64 import b64decode
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_failed_crossing import failed_crossing, restore_images

IMAGE = b"\x89PNG\r\n\x1a\n" + b"pretend frame bytes"
DIGEST = sha256(IMAGE).hexdigest()


def _run_root(directory: Path, *, preserve_image: bool) -> Path:
    run_root = directory / "run"
    (run_root / "worker").mkdir(parents=True)
    if preserve_image:
        (run_root / "model-frame.png").write_bytes(IMAGE)
    body = {
        "model": "google/gemma-4-31b-it",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "observe"},
                    {
                        "type": "input_image",
                        "image_url": {
                            "bytes": len(IMAGE),
                            "kind": "data-image",
                            "media_type": "image/png",
                            "redacted": True,
                            "sha256": DIGEST,
                        },
                    },
                ],
            }
        ],
    }
    record = {
        "kind": "model_crossing",
        "audit": {
            "model_calls": [
                {
                    "outcome": {
                        "kind": "transport_failure",
                        "status_code": 400,
                        "message_chars": 476,
                        "message_sha256": "a" * 64,
                    },
                    "attempts": [
                        {"actual_provider_crossings": [{"request": {"body": body}}]}
                    ],
                }
            ]
        },
    }
    (run_root / "worker" / "process.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return run_root


class ReplayFailedCrossingTests(unittest.TestCase):
    def test_restores_the_redacted_image_by_its_recorded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = _run_root(Path(directory), preserve_image=True)
            body, recorded = failed_crossing(run_root)
            self.assertEqual(recorded["message_chars"], 476)

            rebuilt, restored = restore_images(body, run_root)
            self.assertEqual(restored, 1)
            url = rebuilt["input"][0]["content"][1]["image_url"]
            self.assertIsInstance(url, str)
            self.assertTrue(url.startswith("data:image/png;base64,"))
            self.assertEqual(
                b64decode(url.split(",", 1)[1]),
                IMAGE,
                "restored bytes must be the exact image the stub names",
            )

    def test_refuses_when_no_preserved_image_matches(self) -> None:
        """Replaying without the real image tests a payload never sent.

        That mistake produced a confident, wrong diagnosis of a frozen-code
        defect. Refusing is the whole point of the tool.
        """

        with tempfile.TemporaryDirectory() as directory:
            run_root = _run_root(Path(directory), preserve_image=False)
            body, _recorded = failed_crossing(run_root)
            with self.assertRaisesRegex(SystemExit, "cannot rebuild the request"):
                restore_images(body, run_root)


if __name__ == "__main__":
    unittest.main()
