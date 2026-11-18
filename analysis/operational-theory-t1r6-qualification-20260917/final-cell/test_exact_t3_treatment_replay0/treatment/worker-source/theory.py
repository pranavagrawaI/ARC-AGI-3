"""Mechanical Operational Theory lineage and cycle. No semantic truth oracle."""

import copy
import hashlib
import json
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(encoded(value))
    tmp.replace(path)


class IntegrityError(RuntimeError):
    """Hard stop: evidence, lineage, artifact or ownership integrity failed."""


class FormatError(ValueError):
    """Repairable input; consumes the arm's bounded feedback allowance."""


SCHEMA = json.loads((HERE / "operational-theory.schema.json").read_text())


def validate(theory, parent, evidence, artifacts):
    try:
        jsonschema.Draft202012Validator(SCHEMA).validate(theory)
    except jsonschema.ValidationError as exc:
        raise FormatError(f"{list(exc.absolute_path)}: {exc.message}") from None
    if theory["version"] != (parent["version"] + 1 if parent else 1):
        raise FormatError("version must increment by exactly one")
    if theory["parent_sha256"] != (digest(parent) if parent else None):
        raise FormatError("parent_sha256 must bind current accepted theory")
    refs = theory["evidence_refs"]
    if not refs or any(evidence.get(r["id"]) != r["sha256"] for r in refs):
        raise FormatError(
            "evidence binding must name an available immutable observation"
        )
    known = {r["id"] for r in refs}
    if len(known) != len(refs):
        raise FormatError("duplicate evidence IDs")
    claims = {c["id"]: c for c in theory["claims"]}
    if len(claims) != len(theory["claims"]):
        raise FormatError("duplicate claim IDs")
    for c in claims.values():
        if (
            not set(c["evidence_refs"]) <= known
            or not set(c["counterevidence_refs"]) <= known
        ):
            raise FormatError("claim evidence is absent from canonical evidence_refs")
    procedures = {p["id"]: p for p in theory["procedures"]}
    if len(procedures) != len(theory["procedures"]):
        raise FormatError("duplicate procedure IDs")
    for p in procedures.values():
        if artifacts.get(p["path"]) != p["sha256"]:
            raise FormatError(
                "procedure must bind a registered executable artifact hash"
            )
        if not set(p["claim_refs"]) <= claims.keys():
            raise FormatError("procedure claim reference unknown")
    for bucket in (
        "observations",
        "action_semantics",
        "objectives",
        "uncertainties",
        "experiments",
    ):
        for item in theory[bucket]:
            if (
                not set(item["evidence_refs"]) <= known
                or not set(item["claim_refs"]) <= claims.keys()
            ):
                raise FormatError(f"{bucket}: unknown claim/evidence reference")
    old_claims = {c["id"]: c for c in parent["claims"]} if parent else {}
    if not old_claims.keys() <= claims.keys():
        raise FormatError("claims cannot silently disappear; retain retracted payload")
    changes = {r["claim_id"]: r for r in theory["revisions"]}
    for key, old in old_claims.items():
        if old != claims[key] and key not in changes:
            raise FormatError("changed claim requires explicit revision lineage")
    for r in theory["revisions"]:
        if r["claim_id"] not in old_claims or r["previous_sha256"] != digest(
            old_claims[r["claim_id"]]
        ):
            raise FormatError("revision must bind prior claim payload")
        if not set(r["evidence_refs"]) <= known:
            raise FormatError("revision evidence unknown")
        if r["reason"] == "counterevidence" and not set(r["evidence_refs"]) <= set(
            claims[r["claim_id"]]["counterevidence_refs"]
        ):
            raise FormatError(
                "counterevidence revision must link claim counterevidence"
            )
    if parent and not {r["id"] for r in parent["evidence_refs"]} <= known:
        raise FormatError("canonical evidence cannot silently disappear")


class Store:
    """Host-owned, outside Actor writable roots. Log is the authority; no transcript needed."""

    def __init__(self, root, evidence, artifacts=None, integrity_check=None):
        self.integrity_check = integrity_check
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence = copy.deepcopy(evidence)
        index = self.root / "artifact-index.json"
        self.artifacts = dict(
            artifacts
            if artifacts is not None
            else json.loads(index.read_text())
            if index.exists()
            else {}
        )
        for sha in self.artifacts.values():
            path = self.root / "artifacts" / sha
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sha
            ):
                raise IntegrityError("archived executable artifact hash drift")
        self.log = self.root / "lineage.jsonl"
        self.rows = []
        if self.log.exists():
            self.rows = [json.loads(s) for s in self.log.read_text().splitlines()]
            self._verify()
        self.phase = "idle"
        self.cycle = None
        self.errors = 0
        self.pending = None
        journal = self.root / "cycle.json"
        if journal.exists():
            state = json.loads(journal.read_text())
            self.phase, self.cycle = state["phase"], state["cycle"]
            self.errors, self.pending = state["errors"], state["pending"]

    def register(self, filename, source):
        if (
            not isinstance(filename, str)
            or "/" in filename
            or not filename.endswith(".py")
            or not isinstance(source, str)
        ):
            raise FormatError("invalid executable artifact")
        raw = source.encode()
        sha = hashlib.sha256(raw).hexdigest()
        key = f"helpers/{filename}"
        # References address content versions; changing a helper creates a new ref.
        key = key + "@" + sha
        self.artifacts[key] = sha
        destination = self.root / "artifacts" / sha
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and destination.read_bytes() != raw:
            raise IntegrityError("artifact archive collision")
        destination.write_bytes(raw)
        write(self.root / "artifact-index.json", self.artifacts)
        return {"path": key, "sha256": sha}

    @property
    def current(self):
        return copy.deepcopy(self.rows[-1]["theory"]) if self.rows else None

    def _verify(self):
        parent = None
        for row in self.rows:
            if row["sha256"] != digest(row["theory"]):
                raise IntegrityError("canonical hash drift")
            try:
                validate(row["theory"], parent, self.evidence, self.artifacts)
            except FormatError as exc:
                raise IntegrityError(str(exc)) from exc
            parent = row["theory"]

    def _persist(self):
        write(
            self.root / "cycle.json",
            dict(
                phase=self.phase,
                cycle=self.cycle,
                errors=self.errors,
                pending=self.pending,
            ),
        )

    def _check_evidence(self):
        if self.integrity_check is not None:
            self.integrity_check(self)

    def _publish(self, theory):
        validate(theory, self.current, self.evidence, self.artifacts)
        self._check_evidence()
        row = {"sha256": digest(theory), "theory": copy.deepcopy(theory)}
        with self.log.open("ab") as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n")
            f.flush()
            import os

            os.fsync(f.fileno())
        self.rows.append(row)
        return row["sha256"]

    def feedback(self, message):
        self.errors += 1
        self._persist()
        if self.errors > 3:
            return {
                "status": "INCOMPLETE",
                "reason": "repair allowance exhausted",
                "phase": self.phase,
            }
        return {
            "status": "REPAIR",
            "error": message,
            "phase": self.phase,
            "repairs_remaining": 3 - self.errors,
        }

    def call(self, operation, payload=None):
        self._check_evidence()
        if (
            self.log.exists()
            and [json.loads(s) for s in self.log.read_text().splitlines()] != self.rows
        ):
            raise IntegrityError("canonical log changed outside store")
        if self.errors > 3:
            return {"status": "INCOMPLETE", "reason": "repair allowance exhausted"}
        try:
            result = self._call(operation, payload)
        except (FormatError, KeyError, TypeError) as exc:
            return self.feedback(str(exc))
        self._persist()
        return result

    def _call(self, operation, payload):
        if operation == "read":
            return {
                "theory": self.current,
                "sha256": digest(self.current) if self.current else None,
                "phase": self.phase,
                "cycle": copy.deepcopy(self.cycle),
                "pending": self.pending,
                "evidence_catalog": self.evidence,
                "schema": SCHEMA,
            }
        if operation == "observe":
            if (
                self.phase not in ("idle", "complete")
                or not isinstance(payload, list)
                or not payload
            ):
                raise FormatError(
                    "observe requires idle/complete and nonempty evidence ID list"
                )
            if not set(payload) <= self.evidence.keys():
                raise FormatError("observation unavailable at this frontier")
            self.cycle = {"observed": payload}
            self.phase = "observed"
        elif operation == "reconcile":
            if (
                self.phase != "observed"
                or not isinstance(payload, dict)
                or set(payload) != {"evidence_refs", "claim_refs", "disposition"}
            ):
                raise FormatError(
                    "reconcile after observe: evidence_refs, claim_refs, disposition required"
                )
            if not payload["disposition"] or not set(payload["evidence_refs"]) <= set(
                self.cycle["observed"]
            ):
                raise FormatError("reconciliation must cite observed evidence")
            ids = {c["id"] for c in self.current["claims"]} if self.current else set()
            if not set(payload["claim_refs"]) <= ids or not payload["evidence_refs"]:
                raise FormatError("reconciliation has unknown claims or empty evidence")
            self._check_evidence()
            self.cycle["reconciliation"] = payload
            self.phase = "reconciled"
        elif operation == "revise":
            if self.phase != "reconciled":
                raise FormatError("revise requires observe then reconcile")
            if payload["current_policy"] is not None:
                raise FormatError(
                    "revise sets current_policy=null; decide binds a new prospective record"
                )
            if not set(self.cycle["observed"]) <= {
                r["id"] for r in payload["evidence_refs"]
            }:
                raise FormatError("revision must enter observed evidence")
            self.cycle["theory_sha256"] = self._publish(payload)
            self.phase = "revised"
        elif operation == "select":
            if (
                self.phase != "revised"
                or not isinstance(payload, dict)
                or set(payload) != {"kind", "id"}
            ):
                raise FormatError(
                    "select after revise: kind=uncertainty|procedure and id"
                )
            bucket = {"uncertainty": "uncertainties", "procedure": "procedures"}.get(
                payload["kind"]
            )
            if bucket is None or payload["id"] not in {
                r["id"] for r in self.current[bucket]
            }:
                raise FormatError(
                    "selection must name a canonical active uncertainty or procedure"
                )
            self.cycle["selection"] = payload
            self.phase = "selected"
        elif operation == "declare":
            if self.phase != "selected" or payload not in (
                "explore",
                "exploit",
                "repair",
            ):
                raise FormatError("declare explore/exploit/repair after select")
            self.cycle["mode"] = payload
            self.phase = "declared"
        elif operation == "decide":
            if self.phase != "declared":
                raise FormatError(
                    "decision requires entire observe/reconcile/revise/select/declare cycle"
                )
            try:
                jsonschema.validate(payload, SCHEMA["$defs"]["decision"])
            except jsonschema.ValidationError as exc:
                raise FormatError(exc.message) from None
            if (
                payload["theory_sha256"] != digest(self.current)
                or payload["mode"] != self.cycle["mode"]
                or payload["selection"] != self.cycle["selection"]
            ):
                raise FormatError(
                    "decision must bind active theory, declared mode and selection"
                )
            if not set(payload["claim_refs"]) <= {
                c["id"] for c in self.current["claims"]
            }:
                raise FormatError("decision claim reference unknown")
            if not set(payload["evidence_refs"]) <= {
                r["id"] for r in self.current["evidence_refs"]
            }:
                raise FormatError("decision evidence reference unknown")
            theory = self.current
            theory.update(
                version=theory["version"] + 1,
                parent_sha256=digest(theory),
                current_policy=payload,
                revisions=[],
            )
            self.cycle["decision_sha256"] = self._publish(theory)
            self.phase = "decided"
        elif operation == "prepare":
            if (
                self.phase != "decided"
                or payload != self.current["current_policy"]["actions"]
            ):
                raise FormatError(
                    "prepare must use exact actions from the canonical decision"
                )
            self.pending = {
                "theory_sha256": digest(self.current),
                "actions": payload,
                "cycle": copy.deepcopy(self.cycle),
            }
            with (self.root / "decisions.jsonl").open("a") as f:
                f.write(json.dumps(self.pending, sort_keys=True) + "\n")
            self.phase = "prepared"
        elif operation == "result":
            if self.phase != "prepared":
                raise FormatError("result requires a prepared prospective decision")
            if not isinstance(payload, dict) or set(payload) != {
                "theory",
                "observation_id",
                "relation",
            }:
                raise FormatError("result requires theory, observation_id, relation")
            if payload["relation"] not in (
                "recorded_continuation_not_action_outcome",
                "executed_action_outcome",
            ):
                raise FormatError("declare observation relation explicitly")
            if (
                payload["observation_id"] not in self.evidence
                or payload["observation_id"] in self.cycle["observed"]
            ):
                raise FormatError("result must cite a newly available observation")
            if payload["theory"]["current_policy"] != self.current["current_policy"]:
                raise FormatError(
                    "result preserves prospective decision; revise next cycle to change it"
                )
            if payload["observation_id"] not in {
                r["id"] for r in payload["theory"]["evidence_refs"]
            }:
                raise FormatError("post-observation version must enter result evidence")
            self._publish(payload["theory"])
            self.cycle["result"] = {
                "observation_id": payload["observation_id"],
                "relation": payload["relation"],
                "theory_sha256": digest(self.current),
            }
            with (self.root / "completed.jsonl").open("a") as f:
                f.write(json.dumps(self.cycle, sort_keys=True) + "\n")
            self.phase = "complete"
        else:
            raise FormatError(
                "unknown operation; call workflow_skill() for the protocol"
            )
        return {
            "status": "OK",
            "phase": self.phase,
            "sha256": digest(self.current) if self.current else None,
        }


def replay(root, evidence, artifacts=None):
    store = Store(root, evidence, artifacts)
    decisions = []
    path = Path(root) / "decisions.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        decision = json.loads(line)
        matches = [r for r in store.rows if r["sha256"] == decision["theory_sha256"]]
        if (
            len(matches) != 1
            or matches[0]["theory"]["current_policy"]["actions"] != decision["actions"]
        ):
            raise IntegrityError("decision cannot replay from canonical lineage")
        policy = matches[0]["theory"]["current_policy"]
        source = [r for r in store.rows if r["sha256"] == policy["theory_sha256"]]
        if len(source) != 1:
            raise IntegrityError("prospective policy source missing")
        decisions.append(
            {
                "actions": decision["actions"],
                "policy": policy,
                "source_theory": source[0]["theory"],
            }
        )
    return {
        "decisions": decisions,
        "versions": len(store.rows),
        "transcript_required": False,
    }
