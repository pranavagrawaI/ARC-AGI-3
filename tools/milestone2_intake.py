#!/usr/bin/env python3
"""Pull, inspect and prepare a private Kaggle copy. Never execute or push code."""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse


DUCK = "jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner"
SOURCES = ("dataset_sources", "model_sources", "kernel_sources", "competition_sources")
SETTINGS = ("language", "kernel_type", "enable_gpu", "enable_tpu", "enable_internet",
            "machine_shape", "docker_image", "docker_image_pinning_type")


def reference(value: str) -> str:
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.netloc not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError("Use a https://www.kaggle.com/code/OWNER/SLUG URL or OWNER/SLUG[/VERSION]")
        if url.query or url.fragment:
            raise ValueError("Use OWNER/SLUG/VERSION for a pinned version; URL query/fragment is ambiguous")
        if not url.path.startswith("/code/"):
            raise ValueError("Expected a Kaggle /code/ notebook URL")
        value = url.path[len("/code/"):].rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[1-9][0-9]*)?", value):
        raise ValueError("Expected OWNER/SLUG or OWNER/SLUG/VERSION")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boolean(value: object, field: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field} must be true or false, got {value!r}")


def load_bundle(folder: Path, ref: str) -> tuple[dict, Path, dict | None, list[str]]:
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if meta.get("id", "").lower() != "/".join(ref.split("/")[:2]).lower():
        raise ValueError(f"Metadata identity does not match {ref}")
    if meta.get("language") != "python" or meta.get("kernel_type") not in {"notebook", "script"}:
        raise ValueError("This intake supports Python notebooks and Python scripts")
    code_file = meta.get("code_file")
    if not isinstance(code_file, str) or not code_file:
        raise ValueError("Missing code_file in kernel metadata")
    code = folder / code_file
    if not code.resolve().is_relative_to(folder.resolve()) or not code.is_file():
        raise ValueError("code_file must name an existing file inside the pulled directory")
    for field in SOURCES:
        values = meta.get(field, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"{field} must be a list of strings")
    notebook = None
    if meta["kernel_type"] == "notebook":
        notebook = json.loads(code.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise ValueError("Expected a nbformat 4 notebook with cells")
        cells = []
        for cell in notebook["cells"]:
            source = cell.get("source", "")
            if isinstance(source, list) and all(isinstance(s, str) for s in source):
                source = "".join(source)
            if not isinstance(source, str):
                raise ValueError("Notebook cell source must be text or a list of text")
            if cell.get("cell_type") == "code":
                cells.append(source)
    else:
        cells = [code.read_text(encoding="utf-8")]
    return meta, code, notebook, cells


def acquire(ref: str, target: Path, local: Path | None, cli: str) -> dict:
    if local:
        _, code, _, _ = load_bundle(local, ref)
        target.mkdir()
        shutil.copyfile(local / "kernel-metadata.json", target / "kernel-metadata.json")
        relative = code.relative_to(local)
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code, target / relative)
        return {"reference": ref, "mode": "local", "path": str(local.resolve())}
    command = [cli, "kernels", "pull", ref, "--metadata", "--path", str(target)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    (target.parent / f"{target.name}-pull.log").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"Pull failed for {ref}; see {target.name}-pull.log (no retry made)")
    return {"reference": ref, "mode": "kaggle", "command": command}


def inventory(meta: dict, cells: list[str]) -> dict:
    """Static clues with locations, not a dependency resolver or code evaluator."""
    imports: set[str] = set()
    findings: dict[str, list] = {k: [] for k in
                                 ("install_commands", "input_paths", "urls", "secrets",
                                  "environment_variables", "external_code", "parse_gaps")}
    for index, source in enumerate(cells, 1):
        for line_number, line in enumerate(source.splitlines(), 1):
            location = {"code_cell": index, "line": line_number}
            if re.search(r"\b(pip3?|uv|conda|mamba|apt-get|apt)\s+(?:pip\s+)?install\b", line):
                findings["install_commands"].append({**location, "text": line.strip()})
            for path in re.findall(r"/kaggle/input/[A-Za-z0-9_./*{}-]+", line):
                findings["input_paths"].append({**location, "value": path.rstrip(".")})
            for url in re.findall(r"https?://[^\s\"'<>]+", line):
                findings["urls"].append({**location, "value": url})
            if re.search(r"pickle\.(load|loads)|\b(?:exec|eval)\(|runpy\.|importlib\.|sys\.path|\.sh[\"']", line):
                findings["external_code"].append({**location, "text": line.strip()})
            for name in re.findall(r"get_secret\(\s*['\"]([^'\"]+)['\"]", line):
                findings["secrets"].append({**location, "name": name})
            for name in re.findall(r"(?:getenv\(|environ(?:\.get\(|\[))\s*['\"]([^'\"]+)['\"]", line):
                findings["environment_variables"].append({**location, "name": name})
        # IPython lines are inspected above but cannot be parsed as Python.
        python = "\n".join("" if line.lstrip().startswith(("!", "%")) else line
                           for line in source.splitlines())
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            findings["parse_gaps"].append({"code_cell": index, "reason": "IPython/shell lines"})
        try:
            tree = ast.parse(python)
        except SyntaxError as exc:
            findings["parse_gaps"].append({"code_cell": index, "line": exc.lineno, "reason": exc.msg})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Call):
                text = ast.get_source_segment(python, node) or ""
                # Covers subprocess.run([sys.executable, '-m', 'pip', 'install', ...])
                # and dynamic argument lists, without evaluating any expression.
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "run", "check_call", "check_output", "Popen", "system"}:
                    if re.search(r"\b(?:pip|uv|conda|mamba|apt-get)\b", text) and "install" in text:
                        findings["install_commands"].append(
                            {"code_cell": index, "line": node.lineno, "text": text})
    return {"attachments": {key: meta.get(key, []) for key in SOURCES},
            "runtime": {key: meta.get(key) for key in SETTINGS},
            "imports": sorted(imports), **findings}


def normalized_code(cells: list[str]) -> str:
    return "\n\n".join(f"# --- code cell {i} ---\n{source.rstrip()}"
                         for i, source in enumerate(cells, 1)) + "\n"


def delta(before: dict, after: dict) -> dict:
    return {key: {"duck": before.get(key), "candidate": after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}


def prepare(meta: dict, notebook: dict | None, code: Path, kernel: Path,
            destination: str, accelerator: str | None, internet: str | None) -> tuple[dict, list, list]:
    private = copy.deepcopy(meta)
    private.pop("id_no", None)  # Kaggle prefers the numeric ID over the new slug.
    private.update(id=destination, title=destination.split("/")[1].replace("-", " "),
                   is_private=True, code_file=code.name)
    blockers = []
    repairs = []
    for field in ("enable_gpu", "enable_tpu", "enable_internet"):
        if field not in private:
            blockers.append(f"Missing {field}: source runtime setting is unknown")
            private[field] = False
        else:
            private[field] = boolean(private[field], field)
    if accelerator:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", accelerator):
            raise ValueError("Accelerator must be a Kaggle machine_shape name, e.g. NvidiaRtxPro6000")
        private["machine_shape"] = accelerator
        private["enable_tpu"] = accelerator.lower().startswith("tpu")
        private["enable_gpu"] = accelerator.lower().startswith("nvidia")
        blockers = [b for b in blockers if not b.startswith(("Missing enable_gpu", "Missing enable_tpu"))]
    if internet:
        private["enable_internet"] = internet == "on"
        blockers = [b for b in blockers if not b.startswith("Missing enable_internet")]
    if private["enable_gpu"] and private["enable_tpu"]:
        blockers.append("GPU and TPU are both enabled")
    shape = private.get("machine_shape")
    if (private["enable_gpu"] or private["enable_tpu"]) and not shape:
        blockers.append("Accelerator enabled but machine_shape is missing; use --accelerator")
    if shape and str(shape).lower().startswith("nvidia") and not private["enable_gpu"]:
        blockers.append("Nvidia machine_shape conflicts with enable_gpu=false")
    if shape and str(shape).lower().startswith("tpu") and not private["enable_tpu"]:
        blockers.append("TPU machine_shape conflicts with enable_tpu=false")
    if private.get("docker_image"):
        private["docker_image_pinning_type"] = "original"
    for key in SOURCES:
        private.setdefault(key, [])
    kernel.mkdir()
    if notebook is None:
        shutil.copyfile(code, kernel / code.name)
    else:
        notebook = copy.deepcopy(notebook)
        if notebook.get("nbformat_minor", 0) < 5:
            notebook["nbformat_minor"] = 5
            repairs.append("Upgrade notebook format to 4.5 for cell IDs")
        metadata = notebook.setdefault("metadata", {})
        wanted = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if metadata.get("kernelspec") != wanted:
            metadata["kernelspec"] = wanted
            repairs.append("Set Python 3 kernelspec for Papermill")
        language = metadata.setdefault("language_info", {})
        if language.get("name") != "python":
            language["name"] = "python"
            repairs.append("Set language_info.name=python")
        seen = set()
        for i, cell in enumerate(notebook["cells"]):
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) or cell_id in seen:
                cell_id = f"intake-cell-{i}"
                while cell_id in seen:
                    cell_id += "x"
                cell["id"] = cell_id
                repairs.append(f"Set unique ID for notebook cell {i}")
            seen.add(cell_id)
        write_json(kernel / code.name, notebook)
    write_json(kernel / "kernel-metadata.json", private)
    return private, repairs, blockers


def render_report(report: dict) -> str:
    inv = report["candidate_inventory"]
    lines = ["# Milestone 2 notebook intake", "", f"Status: **{report['status']}**", "",
             f"Candidate: `{report['candidate']['reference']}`",
             f"Duck: `{report['duck']['reference']}`",
             f"Private destination: `{report['private_id']}`", "",
             "## Requirements", ""]
    for kind, values in inv["attachments"].items():
        lines.append(f"- {kind}: " + (", ".join(f"`{v}`" for v in values) or "none declared"))
    lines += ["", "Runtime settings:", "", "```json", json.dumps(inv["runtime"], indent=2), "```", "",
              "Python imports (module names, not a pip requirements file): " + ", ".join(inv["imports"]), ""]
    for key in ("install_commands", "input_paths", "urls", "secrets", "environment_variables", "external_code", "parse_gaps"):
        lines += [f"### {key.replace('_', ' ').capitalize()}", ""]
        for item in inv[key]:
            lines += [f"Code cell {item['code_cell']}, line {item.get('line', '?')}:", "", "```text",
                      str(item.get("text", item.get("value", item.get("name", item.get("reason"))))), "```", ""]
        if not inv[key]:
            lines += ["None detected.", ""]
    lines += ["## Differences from Duck", "", "Executable cell diff: `code.diff`; configuration diff: `settings.diff`.",
              "Notebook outputs and execution counts are excluded from the code diff.", "", "```json",
              json.dumps(report["differences"], indent=2), "```", "",
              "## Private copy changes", "", "```json", json.dumps(report["private_changes"], indent=2), "```", ""]
    lines += [f"- {repair}" for repair in report["notebook_repairs"]]
    lines += ["", "## Unresolved checks", ""]
    lines += [f"- {item}" for item in report["blockers"] + report["limitations"]]
    if report["commit_command"]:
        lines += ["", "## Commit command (not executed)", "", "```bash", report["commit_command"], "```",
                  "", "A push creates and runs a Kaggle version. Review the report, input access and quota before authorizing it."]
    return "\n".join(lines) + "\n"


def owner_default() -> str | None:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
    if config.is_file():
        return json.loads(config.read_text(encoding="utf-8")).get("username")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="Public OWNER/SLUG[/VERSION] or Kaggle /code/ URL")
    parser.add_argument("--duck", default=DUCK, help=f"Comparison baseline (default: {DUCK})")
    parser.add_argument("--owner", help="Your Kaggle username; defaults to KAGGLE_USERNAME or kaggle.json")
    parser.add_argument("--slug", help="New private kernel slug; default includes a unique timestamp")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--source-dir", type=Path, help="Use a previously pulled candidate directory (offline)")
    parser.add_argument("--duck-dir", type=Path, help="Use a previously pulled Duck directory (offline)")
    parser.add_argument("--accelerator", help="Override Kaggle machine_shape; otherwise preserve donor setting")
    parser.add_argument("--internet", choices=("on", "off"), help="Override donor internet setting explicitly")
    parser.add_argument("--timeout", type=int, help="Session seconds for the printed commit command; no default override")
    parser.add_argument("--kaggle", default="kaggle", help="Kaggle CLI executable (tested with 2.2.4)")
    args = parser.parse_args(argv)
    output = None
    try:
        candidate_ref, duck_ref = reference(args.notebook), reference(args.duck)
        owner = args.owner or owner_default()
        if not owner or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
            raise ValueError("Supply --owner YOUR_KAGGLE_USERNAME (or set KAGGLE_USERNAME)")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        stem = re.sub(r"[^a-z0-9]+", "-", candidate_ref.split('/')[1].lower())[:20].strip("-")
        slug = args.slug or f"m2-{stem}-{stamp}"
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or not 5 <= len(slug) <= 100:
            raise ValueError("Use a 5-100 character lowercase, dash-separated --slug")
        destination = f"{owner}/{slug}"
        if destination.lower() in {"/".join(r.split("/")[:2]).lower() for r in (candidate_ref, duck_ref)}:
            raise ValueError("Private destination must differ from the candidate and Duck IDs")
        if args.timeout is not None and args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        candidate_output = (args.output or Path("runs/milestone2-intake") / slug).resolve()
        candidate_output.mkdir(parents=True, exist_ok=False)
        output = candidate_output
        cli_version = "not used (offline)"
        if not args.source_dir or not args.duck_dir:
            cli_version = subprocess.run([args.kaggle, "--version"], check=True, capture_output=True,
                                         text=True, timeout=30).stdout.strip()
        candidate_receipt = acquire(candidate_ref, output / "candidate", args.source_dir, args.kaggle)
        duck_receipt = acquire(duck_ref, output / "duck", args.duck_dir, args.kaggle)
        cm, cp, cn, cc = load_bundle(output / "candidate", candidate_ref)
        dm, _, _, dc = load_bundle(output / "duck", duck_ref)
        if boolean(cm.get("is_private"), "is_private"):
            raise ValueError("Candidate is private; this tool expects a public notebook")
        ci, di = inventory(cm, cc), inventory(dm, dc)
        private, repairs, blockers = prepare(cm, cn, cp, output / "kernel", destination,
                                             args.accelerator, args.internet)
        for name, before, after in (("code", normalized_code(dc), normalized_code(cc)),
                                    ("settings", json.dumps({k: dm.get(k) for k in SETTINGS + SOURCES}, indent=2) + "\n",
                                     json.dumps({k: cm.get(k) for k in SETTINGS + SOURCES}, indent=2) + "\n")):
            (output / f"{name}.diff").write_text("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"duck/{name}", tofile=f"candidate/{name}")), encoding="utf-8")
        differences = {
            "settings": delta(di["runtime"], ci["runtime"]),
            "attachments": {k: {"added": sorted(set(ci["attachments"][k]) - set(di["attachments"][k])),
                                "removed": sorted(set(di["attachments"][k]) - set(ci["attachments"][k]))} for k in SOURCES},
            "imports_added": sorted(set(ci["imports"]) - set(di["imports"])),
            "imports_removed": sorted(set(di["imports"]) - set(ci["imports"])),
            "code_identical": normalized_code(cc) == normalized_code(dc),
        }
        limitations = [
            "Static notebook inspection only. Attached solver code, wheels, weights, pickles and transitive dependencies were not downloaded or executed; this is not a full solver diff.",
            "Attachment accessibility, competition consent, secrets, available GPU/quota, and successful execution remain unverified.",
            "Dataset references returned by Kaggle pull may be unversioned. Preserved bytes/hashes pin the notebooks, not all input contents.",
        ]
        if not private.get("docker_image"):
            limitations.append("No docker_image was supplied: Kaggle will select the runtime image.")
        if ci["parse_gaps"]:
            limitations.append("Some cells could not be fully parsed; inspect parse_gaps and the original source.")
        command = [args.kaggle, "kernels", "push", "--path", str(output / "kernel")]
        if private.get("machine_shape"):
            command += ["--accelerator", str(private["machine_shape"])]
        if args.timeout:
            command += ["--timeout", str(args.timeout)]
        report = {"schema_version": 1, "tool_sha256": digest(Path(__file__)),
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": "blocked" if blockers else "prepared_unexecuted", "kaggle_cli": cli_version,
                  "candidate": candidate_receipt, "duck": duck_receipt, "private_id": destination,
                  "candidate_inventory": ci, "duck_inventory": di, "differences": differences,
                  "private_changes": {k: {"original": cm.get(k), "private": private.get(k)}
                                      for k in sorted(cm.keys() | private.keys()) if cm.get(k) != private.get(k)},
                  "notebook_repairs": repairs, "blockers": blockers, "limitations": limitations,
                  "commit_command": None if blockers else shlex.join(command),
                  "sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
        write_json(output / "intake.json", report)
        (output / "REPORT.md").write_text(render_report(report), encoding="utf-8")
        print(f"{report['status']}: {output / 'REPORT.md'}")
        print(f"Private config: {output / 'kernel/kernel-metadata.json'}")
        print("No code executed, kernel pushed, or competition submission made.")
        return 2 if blockers else 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if output is not None:
            write_json(output / "ERROR.json", {"error": str(exc)})
        print(f"Intake failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
