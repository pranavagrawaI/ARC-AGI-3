from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.builder_matched_replay import execute_zero_arc_builder_replay
from tycho_nooa.builder_proposal_adapters import (
    CODEACT_MECHANICS,
    COMMON_BUILDER_TASK,
    NATIVE_MECHANICS,
    BuilderCodeActController,
    BuilderNativeController,
    BuilderReportReturn,
)
from tycho_nooa.builder_proposal_protocol import (
    BuilderEvaluationCapability,
    FrozenBuildCase,
    ProposalLedger,
    canonicalize_source,
)
from tycho_nooa.builder_sterile_execution import (
    BuilderExecutionManifest,
    BuilderSterileExecutor,
)
from tycho_nooa.execution_events import (
    ExecutionProvenanceLog,
    InvocationProvenance,
)
from tycho_nooa.theory_evaluation import (
    EvaluationRunStatus,
    EvidenceFrontier,
    ObligationKind,
    ObligationResult,
    ObligationStatus,
    PublicationDisposition,
    TheoryEvaluation,
)
from tycho_nooa.theory_transaction import TheoryTransaction


class FixtureEvaluator:
    evaluator_sha256 = "fixture-evaluator"

    def __init__(self, results: dict[bytes, tuple[ObligationStatus, ...]]) -> None:
        self.results = {
            canonicalize_source(source): statuses for source, statuses in results.items()
        }

    def evaluate(self, model: bytes, frontier: EvidenceFrontier) -> TheoryEvaluation:
        return TheoryEvaluation(
            model_sha256=hashlib.sha256(model).hexdigest(),
            evidence_frontier_sha256=frontier.sha256,
            evaluator_sha256=self.evaluator_sha256,
            run_status=EvaluationRunStatus.COMPLETE,
            obligations=tuple(
                ObligationResult(
                    obligation_id=f"T{index}",
                    kind=ObligationKind.TRANSITION,
                    status=status,
                    description=f"transition {index}",
                )
                for index, status in enumerate(self.results[model], 1)
            ),
        )


async def test_sterile_builder_realm_persists_locals_with_one_typed_egress(
    tmp_path: Path,
) -> None:
    transaction, case, ledger = proposal_fixture(tmp_path)
    manifest = BuilderExecutionManifest(
        invocation_id="builder-sterile-1",
        capabilities=("builder.evaluate",),
        input_values={"case": case.builder_view()},
    )

    async with BuilderSterileExecutor(
        manifest,
        return_type=BuilderReportReturn,
        capability_handlers={
            "builder.evaluate": BuilderEvaluationCapability(ledger)
        },
    ) as executor:
        first = await executor.run_cell(
            """
parent = case['incumbent']['source']
candidate = 'repair\\n'
receipt = builder.evaluate(
    candidate_source=candidate,
    parent_source=parent,
    declared_parent_source_sha256=case['incumbent']['source_sha256'],
    provenance={'reason': 'repair T1'},
)
open('scratch.txt', 'w').write('realm local')
"""
        )
        second = await executor.run_cell(
            """
unauthorized = False
try:
    builder.evaluate._client.call('evidence.query', {})
except Exception:
    unauthorized = True
immutable = False
try:
    case['incumbent']['source'] = 'tampered'
except TypeError:
    immutable = True
assert unauthorized and immutable and candidate == 'repair\\n'
return_result(
    case_sha256=case['case_sha256'],
    nominated_candidate_source_sha256=receipt['candidate_source_sha256'],
    judgment='repair is Pareto-safe',
    evidence_references=[],
    open_uncertainties=[],
)
"""
        )

    assert first.status == "complete"
    assert "candidate" in second.locals_preview
    assert second.return_value is not None
    assert len(ledger.receipts) == 1
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"
    report = second.return_value.to_protocol(case)
    decision = transaction.finish(report=report, receipts=ledger)
    assert decision.disposition is PublicationDisposition.PUBLISHABLE
    assert (transaction.workspace / "world_model.py").read_bytes() == b"repair\n"


async def test_codeact_adapter_runs_the_provider_two_tool_lifecycle(
    tmp_path: Path,
) -> None:
    transaction, case, ledger = proposal_fixture(tmp_path)
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "execute_python",
                {
                    "code": (
                        "parent = case['incumbent']['source']\n"
                        "candidate = 'repair\\n'\n"
                        "receipt = builder.evaluate("
                        "candidate_source=candidate, parent_source=parent, "
                        "declared_parent_source_sha256="
                        "case['incumbent']['source_sha256'], "
                        "provenance={'reason': 'repair T1'})"
                    )
                },
            ),
            response(
                "return_result",
                {
                    "case_sha256": case.case_sha256,
                    "nominated_candidate_source_sha256": (
                        "sha256:" + hashlib.sha256(b"repair\n").hexdigest()
                    ),
                    "judgment": "repair is Pareto-safe",
                    "evidence_references": [],
                    "open_uncertainties": [],
                },
            ),
        ]
    )
    log = ExecutionProvenanceLog(
        tmp_path / "codeact.jsonl", actor_session_id="builder-shadow"
    )
    controller = BuilderCodeActController(
        llm=llm,
        case=case,
        ledger=ledger,
        provenance=InvocationProvenance(log, invocation_id="builder-codeact-1"),
    )

    report = await controller.run()

    assert report.nominated_candidate_source_sha256 == (
        "sha256:" + hashlib.sha256(b"repair\n").hexdigest()
    )
    assert len(controller.provider_turns) == 2
    assert len(ledger.receipts) == 1
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"
    assert controller.provider_turns[0].messages[0]["content"] == (
        f"{COMMON_BUILDER_TASK}\n\n{CODEACT_MECHANICS}"
    )
    provider_case = json.loads(controller.provider_turns[0].messages[1]["content"])
    assert provider_case == {"case": case.prompt_preview()}
    assert len(controller.provider_turns[0].messages[1]["content"].encode()) < 16_000


async def test_native_adapter_uses_the_same_case_evaluator_and_report_contract(
    tmp_path: Path,
) -> None:
    transaction, case, ledger = proposal_fixture(tmp_path)
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "run_python",
                {
                    "code": "candidate = 'repair\\n'; print(candidate)",
                    "values": {},
                },
            ),
            response(
                "evaluate_candidate",
                {
                    "candidate_source": "repair\n",
                    "parent_source": "root\n",
                    "declared_parent_source_sha256": case.incumbent_source_sha256,
                    "provenance": {"reason": "repair T1"},
                },
            ),
            response(
                "return_result",
                {
                    "case_sha256": case.case_sha256,
                    "nominated_candidate_source_sha256": (
                        "sha256:" + hashlib.sha256(b"repair\n").hexdigest()
                    ),
                    "judgment": "repair is Pareto-safe",
                    "evidence_references": [],
                    "open_uncertainties": [],
                },
            ),
        ]
    )
    controller = BuilderNativeController(llm=llm, case=case, ledger=ledger)

    report = await controller.run()

    assert report.case_sha256 == case.case_sha256
    assert controller.python_calls == 1
    assert len(controller.provider_usage) == 3
    assert len(ledger.receipts) == 1
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"
    system = controller._messages[0]["content"]
    assert system == f"{COMMON_BUILDER_TASK}\n\n{NATIVE_MECHANICS}"
    assert json.loads(controller._messages[1]["content"]) == {
        "case": case.prompt_preview()
    }


async def test_zero_arc_matched_replay_records_the_same_semantic_candidate(
    tmp_path: Path,
) -> None:
    transaction, case, _ = proposal_fixture(tmp_path)
    candidate_sha = "sha256:" + hashlib.sha256(b"repair\n").hexdigest()
    native_llm = FakeLLMClient(
        scripted_responses=[
            response(
                "evaluate_candidate",
                {
                    "candidate_source": "repair\n",
                    "parent_source": "root\n",
                    "declared_parent_source_sha256": case.incumbent_source_sha256,
                    "provenance": {"reason": "repair T1"},
                },
            ),
            response(
                "return_result",
                {
                    "case_sha256": case.case_sha256,
                    "nominated_candidate_source_sha256": candidate_sha,
                    "judgment": "repair is Pareto-safe",
                    "evidence_references": [],
                    "open_uncertainties": [],
                },
            ),
        ]
    )
    codeact_llm = FakeLLMClient(
        scripted_responses=[
            response(
                "execute_python",
                {
                    "code": (
                        "parent = case['incumbent']['source']\n"
                        "candidate = 'repair\\n'\n"
                        "receipt = builder.evaluate("
                        "candidate_source=candidate, parent_source=parent, "
                        "declared_parent_source_sha256="
                        "case['incumbent']['source_sha256'], provenance={})"
                    )
                },
            ),
            response(
                "return_result",
                {
                    "case_sha256": case.case_sha256,
                    "nominated_candidate_source_sha256": candidate_sha,
                    "judgment": "repair is Pareto-safe",
                    "evidence_references": [],
                    "open_uncertainties": [],
                },
            ),
        ]
    )

    report = await execute_zero_arc_builder_replay(
        case=case,
        evaluator_factory=lambda: FixtureEvaluator(
            {
                b"root\n": (ObligationStatus.FAILED, ObligationStatus.PASSED),
                b"repair\n": (ObligationStatus.PASSED, ObligationStatus.PASSED),
            }
        ),
        provider_settings={
            "model": "fixture",
            "provider_route": "local-fake",
            "sampling": {"temperature": 0},
        },
        native_llm=native_llm,
        codeact_llm=codeact_llm,
        destination=tmp_path / "matched",
    )

    assert report.native.case_sha256 == report.codeact.case_sha256
    assert report.native.nominated_candidate_source_sha256 == candidate_sha
    assert report.codeact.nominated_candidate_source_sha256 == candidate_sha
    assert report.native.semantic_outcomes == report.codeact.semantic_outcomes
    assert report.native.metrics.first_pareto_safe_provider_call == 1
    assert report.codeact.metrics.first_pareto_safe_provider_call == 1
    assert report.native.metrics.total_cost_to_first_pareto_safe is None
    assert report.codeact.metrics.total_cost_to_first_pareto_safe is None
    assert (transaction.workspace / "world_model.py").read_bytes() == b"root\n"
    assert (tmp_path / "matched" / "report.json").is_file()


def proposal_fixture(
    tmp_path: Path,
) -> tuple[TheoryTransaction, FrozenBuildCase, ProposalLedger]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "world_model.py").write_bytes(b"root\n")
    level = workspace / "level_0"
    level.mkdir()
    (level / "turn_000.txt").write_text("observation\n")
    evaluator = FixtureEvaluator(
        {
            b"root\n": (ObligationStatus.FAILED, ObligationStatus.PASSED),
            b"repair\n": (ObligationStatus.PASSED, ObligationStatus.PASSED),
        }
    )
    transaction = TheoryTransaction.begin(
        workspace=workspace,
        evaluator=evaluator,
        frontier=EvidenceFrontier.freeze(workspace),
        cause="adapter test",
        store_root=tmp_path / "transaction",
    )
    case = transaction.freeze_build_case(metadata={"task": "repair"})
    ledger = ProposalLedger(case, evaluator, tmp_path / "proposal")
    return transaction, case, ledger


def response(name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=f"{name}-call", name=name, arguments=json.dumps(arguments))
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )
