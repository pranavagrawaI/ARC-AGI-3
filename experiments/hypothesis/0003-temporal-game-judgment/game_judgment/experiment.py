"""Execution engine for the frozen 0003 evidence ladder."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.request import Request, urlopen

from .contract import ContractError, Decision, parse_decision
from .provider import SYSTEM_PROMPT, assemble_provider_request, normalize_response
from .requests import ARMS
from .scoring import score
from .serialization import canonical_bytes, digest


class RunError(RuntimeError):
    pass


class Transport(Protocol):
    def call(self, key: dict[str, Any], request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...


@dataclass
class HTTPTransport:
    endpoint: str
    api_key: str
    timeout: int

    def call(self, key: dict[str, Any], request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        body = canonical_bytes(request)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = perf_counter()
        http_request = Request(self.endpoint, data=body, headers=headers)
        with urlopen(http_request, timeout=self.timeout) as response:
            response_body = response.read()
            response_headers = dict(response.headers.items())
            status = response.status
        elapsed = perf_counter() - started
        return json.loads(response_body), {
            "http_status": status,
            "headers": response_headers,
            "latency_seconds": elapsed,
            "request_bytes": len(body),
            "response_bytes": len(response_body),
        }


@dataclass
class ReplayTransport:
    payloads: dict[tuple[Any, ...], tuple[str, dict[str, Any]]]

    @classmethod
    def load(cls, path: Path) -> "ReplayTransport":
        payloads: dict[tuple[Any, ...], tuple[str, dict[str, Any]]] = {}
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["phase"],
                row["case_id"],
                row["arm"],
                row.get("repeat", 0),
                row.get("variant", "intact"),
                row.get("sequence_index", 0),
            )
            if key in payloads:
                raise RunError(f"duplicate replay response at line {line_number}: {key}")
            request_sha256 = row.get("request_sha256")
            if (
                not isinstance(request_sha256, str)
                or len(request_sha256) != 64
                or any(character not in "0123456789abcdef" for character in request_sha256)
            ):
                raise RunError(
                    f"missing or invalid request_sha256 at replay line {line_number}"
                )
            payloads[key] = request_sha256, row["response"]
        return cls(payloads)

    def call(self, key: dict[str, Any], request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        lookup = (
            key["phase"], key["case_id"], key["arm"], key.get("repeat", 0),
            key.get("variant", "intact"), key.get("sequence_index", 0),
        )
        try:
            expected_request_sha256, payload = self.payloads[lookup]
        except KeyError as exc:
            raise RunError(f"missing replay response: {lookup}") from exc
        actual_request_sha256 = digest(request)
        if actual_request_sha256 != expected_request_sha256:
            raise RunError(
                "replay request mismatch for "
                f"{lookup}: expected {expected_request_sha256}, got {actual_request_sha256}"
            )
        return deepcopy(payload), {
            "transport": "replay",
            "latency_seconds": 0.0,
            "request_sha256": actual_request_sha256,
        }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def audit_openrouter_endpoint(config: dict[str, Any]) -> dict[str, Any]:
    with urlopen(config["endpoint_audit"], timeout=30) as response:
        payload = json.loads(response.read())
    expected = config["provider_audit"]
    matches = [
        row for row in payload["data"]["endpoints"]
        if row.get("provider_name") == expected["provider_name"]
        and row.get("tag") == expected["tag"]
        and row.get("quantization") == expected["quantization"]
        and config["model_revision"] in row.get("name", "")
    ]
    if len(matches) != 1:
        raise RunError("frozen primary provider endpoint is unavailable or ambiguous")
    endpoint = matches[0]
    missing = sorted(set(expected["required_parameters"]) - set(endpoint["supported_parameters"]))
    if missing:
        raise RunError(f"primary provider endpoint lacks required parameters: {missing}")
    if endpoint.get("status") != 0:
        raise RunError(f"primary provider endpoint status is not healthy: {endpoint.get('status')}")
    return {
        "schema": "0003-provider-audit-v1",
        "source": config["endpoint_audit"],
        "source_sha256": digest(payload),
        "endpoint": endpoint,
    }


def audit_openai_compatible_endpoint(endpoint: str, model_path: str, model: str) -> dict[str, Any]:
    url = endpoint.rstrip("/") + model_path
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read())
    ids = {row.get("id") for row in payload.get("data", [])}
    if model not in ids:
        raise RunError(f"transfer model {model!r} is not served by {url}")
    return {"schema": "0003-transfer-provider-audit-v1", "source": url, "source_sha256": digest(payload), "models": sorted(ids)}


def thinking_config(config: dict[str, Any], enabled: bool) -> dict[str, Any]:
    value = deepcopy(config)
    value["reasoning"] = {"enabled": enabled, "exclude": False}
    prompt = value.get("system_prompt", SYSTEM_PROMPT)
    prompt = prompt.removeprefix("<|think|>\n")
    if enabled:
        prompt = "<|think|>\n" + prompt
    value["system_prompt"] = prompt
    return value


def perturb_judgment_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    perturbed = deepcopy(case)
    if perturbed["case_id"] != "game-judgment" or len(perturbed["discriminators"]) != 1:
        raise RunError("frozen perturbation target is unavailable")
    discriminator = perturbed["discriminators"][0]
    theories = discriminator["distinguishes"]
    first = deepcopy(discriminator["predicted_outcomes"][theories[0]])
    before = deepcopy(discriminator["predicted_outcomes"][theories[1]])
    discriminator["predicted_outcomes"][theories[1]] = first
    return perturbed, {
        "schema": "0003-perturbation-v1",
        "target": f"discriminators/{discriminator['discriminator_id']}/predicted_outcomes/{theories[1]}",
        "before": before,
        "after": first,
        "unchanged": ["ids", "ordering", "prose", "legal_actions", "protected_successor"],
        "expected_intact_signature": ["PROBE", discriminator["discriminator_id"], discriminator["action"]],
        "expected_perturbed_signature": ["ABSTAIN", "insufficient_evidence"],
    }


class ExperimentRunner:
    def __init__(
        self,
        root: Path,
        output: Path,
        config: dict[str, Any],
        primary: Transport,
        transfer: Transport | None = None,
    ) -> None:
        self.root = root
        self.output = output
        self.config = config
        self.primary = primary
        self.transfer = transfer
        self.batch = json.loads((root / "fixtures/batch.json").read_text())
        self.labels = {
            row["case_id"]: row
            for row in json.loads((root / "scorer/labels.json").read_text())["labels"]
        }
        self.results: list[dict[str, Any]] = []
        self.request_count = 0
        self.provider_call_count = 0
        self.cost_usd = 0.0

    def _call(
        self,
        case: dict[str, Any],
        arm: str,
        phase: str,
        repeat: int,
        *,
        variant: str = "intact",
        sequence_index: int = 0,
        request_config: dict[str, Any] | None = None,
        transport: Transport | None = None,
        score_case: dict[str, Any] | None = None,
        score_response: bool = True,
    ) -> dict[str, Any]:
        if self.request_count >= int(self.config["max_model_requests"]):
            raise RunError("model request cap reached")
        key = {
            "phase": phase,
            "case_id": case["case_id"],
            "arm": arm,
            "repeat": repeat,
            "variant": variant,
            "sequence_index": sequence_index,
        }
        request, request_audit = assemble_provider_request(
            case, arm, self.root, request_config or self.config
        )
        index = self.request_count
        _write(self.output / "requests" / f"{index:04d}.json", {"key": key, "request": request, "audit": request_audit})
        self.request_count += 1
        payload, transport_audit = (transport or self.primary).call(key, request)
        replay = transport_audit.get("transport") == "replay"
        if not replay:
            self.provider_call_count += 1
        _write(self.output / "responses" / f"{index:04d}.json", payload)
        usage = payload.get("usage", {})
        cost = usage.get("cost", 0.0)
        if type(cost) not in (int, float):
            cost = 0.0
        if not replay:
            self.cost_usd += float(cost)
        if self.cost_usd > float(self.config["max_cost_usd"]):
            raise RunError("provider cost cap exceeded")
        row: dict[str, Any] = {
            **key,
            "request_sha256": request_audit["provider_request_sha256"],
            "transport": transport_audit,
            "raw_response_sha256": digest(payload),
            "usage": usage,
        }
        try:
            normalized = normalize_response(payload)
            decision = parse_decision(normalized["decision"])
            row["provider"] = normalized["provider"]
            row["model"] = normalized["model"]
            if normalized["model"] != request["model"]:
                raise RunError(
                    f"model drift: expected {request['model']}, got {normalized['model']}"
                )
            row["finish_reason"] = normalized["finish_reason"]
            row["reasoning_returned"] = normalized["reasoning"] is not None
            expected_provider = self.config.get("provider_audit", {}).get("provider_name")
            if phase != "transfer" and normalized["provider"] is not None and expected_provider and normalized["provider"] != expected_provider:
                raise RunError(
                    f"provider drift: expected {expected_provider}, got {normalized['provider']}"
                )
            row["decision"] = {"tool": decision.tool, "arguments": decision.arguments}
            row["signature"] = list(decision.signature)
            if score_response:
                authority_case = score_case or case
                row["score"] = score(
                    authority_case, self.labels[authority_case["case_id"]], decision
                )
        except (ValueError, ContractError, json.JSONDecodeError, RunError) as exc:
            row["parser_error"] = str(exc)
            if score_response:
                row["score"] = {"contract": False, "pass": False}
        self.results.append(row)
        return row

    @staticmethod
    def _stable_success(rows: list[dict[str, Any]]) -> bool:
        if not rows or not all(row.get("score", {}).get("pass") for row in rows):
            return False
        by_case: dict[str, set[tuple[Any, ...]]] = {}
        for row in rows:
            by_case.setdefault(row["case_id"], set()).add(tuple(row.get("signature", [])))
        return all(len(signatures) == 1 for signatures in by_case.values())

    def run_ladder(self) -> str:
        repetitions = int(self.config["repetitions"])
        summaries = []
        for arm in ARMS:
            rows = [
                self._call(case, arm, "ladder", repeat)
                for case in self.batch["cases"]
                for repeat in range(repetitions)
            ]
            success = self._stable_success(rows)
            summaries.append({"arm": arm, "success": success, "passes": sum(row.get("score", {}).get("pass", False) for row in rows), "total": len(rows)})
            if success:
                _write(self.output / "ladder-summary.json", {"arms": summaries, "minimal_successful_arm": arm})
                return arm
        _write(self.output / "ladder-summary.json", {"arms": summaries, "minimal_successful_arm": None})
        raise RunError("no arm passed with repeated signature stability")

    def run_thinking_sequence(self, arm: str) -> dict[str, Any]:
        rows = []
        for sequence_index, enabled in enumerate(self.config["thinking_sequence"]):
            condition = thinking_config(self.config, bool(enabled))
            for case in self.batch["cases"]:
                rows.append(
                    self._call(
                        case, arm, "thinking", 0,
                        sequence_index=sequence_index,
                        variant="thinking-on" if enabled else "thinking-off",
                        request_config=condition,
                    )
                )
        first = {row["case_id"]: row.get("signature") for row in rows if row["sequence_index"] == 0}
        last = {row["case_id"]: row.get("signature") for row in rows if row["sequence_index"] == len(self.config["thinking_sequence"]) - 1}
        summary = {"first_off_equals_last_off": first == last, "all_contract_pass": all(row.get("score", {}).get("pass") for row in rows), "generated_reasoning_is_diagnostic_only": True}
        _write(self.output / "thinking-summary.json", summary)
        return summary

    def run_perturbation(self, arm: str) -> dict[str, Any]:
        if arm not in {"F", "G"}:
            summary = {"eligible": False, "reason": "minimal successful arm is not F or G"}
            _write(self.output / "perturbation-summary.json", summary)
            return summary
        intact = next(case for case in self.batch["cases"] if case["case_id"] == "game-judgment")
        perturbed, intervention = perturb_judgment_case(intact)
        ladder_signatures = {
            tuple(row.get("signature", [])) for row in self.results
            if row["phase"] == "ladder" and row["case_id"] == "game-judgment" and row["arm"] == arm
        }
        expected_intact = tuple(intervention["expected_intact_signature"])
        if ladder_signatures != {expected_intact}:
            summary = {"eligible": False, "reason": "intact ladder decision did not use the frozen discriminator", "intervention": intervention}
            _write(self.output / "perturbation-summary.json", summary)
            return summary
        rows = []
        for sequence_index, variant in enumerate(self.config["perturbation_sequence"]):
            target = perturbed if variant == "perturbed" else intact
            for repeat in range(int(self.config["repetitions"])):
                rows.append(
                    self._call(
                        target, arm, "perturbation", repeat,
                        variant=variant,
                        sequence_index=sequence_index,
                        score_response=False,
                    )
                )
        expected_perturbed = tuple(intervention["expected_perturbed_signature"])
        passed = all(
            tuple(row.get("signature", [])) == (expected_perturbed if row["variant"] == "perturbed" else expected_intact)
            for row in rows
        )
        summary = {"eligible": True, "pass": passed, "intervention": intervention}
        _write(self.output / "perturbation-summary.json", summary)
        return summary

    def run_transfer(self, arm: str, transfer_config: dict[str, Any]) -> dict[str, Any]:
        if self.transfer is None:
            raise RunError("required E4B transfer transport is unavailable")
        rows = [
            self._call(
                case, arm, "transfer", repeat,
                request_config=transfer_config,
                transport=self.transfer,
            )
            for case in self.batch["cases"]
            for repeat in range(int(self.config["repetitions"]))
        ]
        summary = {"model": transfer_config["model"], "stable_success": self._stable_success(rows)}
        _write(self.output / "transfer-summary.json", summary)
        return summary

    def finish(
        self, status: str, error: str | None = None, *, transport_mode: str = "provider"
    ) -> None:
        _write(
            self.output / "results.json",
            {
                "schema": "0003-experiment-results-v2",
                "status": status,
                "error": error,
                "transport_mode": transport_mode,
                "environment_actions": 0,
                "model_requests": self.request_count,
                "provider_calls": self.provider_call_count,
                "cost_usd": self.cost_usd,
                "results": self.results,
            },
        )
