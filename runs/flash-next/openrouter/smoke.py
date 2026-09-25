"""Run an arm's shipped source tree against OpenRouter qwen/qwen3.8-flash (= Qwen/Qwen3.8-Flash-Next).

Mirrors the Kaggle notebook: bundle src roots first on sys.path (editable installs dropped), the arm's
env applied before the benchmark pickle imports tool_agent, then bm.run on offline public games.

Usage: python -B smoke.py ARM OUT_DIR GAME [GAME ...] [--minutes M] [--concurrency C]
  ARM = k | kf   (k uses Keith's bundle at /tmp/keith, kf our uploaded source-dataset)
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLES = {"k": Path("/tmp/keith"), "kf": HERE.parent / "kf" / "source-dataset"}
ENV_DIR = Path.home() / "projects/arc-agi-3-reference/competition-env/environment_files"
# Keith's serving_setup.py analyzer block, with the endpoint swapped to OpenRouter.
ANALYZER_ENV = {
    "LOCAL_ANALYZER_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
    "LOCAL_ANALYZER_PROVIDER": "openrouter",
    "OPENAI_PROVIDER": "openrouter",
    "LOCAL_ANALYZER_MODEL_ID": "qwen/qwen3.8-flash",
    "INFERENCE_ANALYZER_MODEL": "qwen/qwen3.8-flash",
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
    "ONLY_RESET_LEVELS": "true",
    "TAAF_MINIMAL_DIAGNOSTICS": "1",
    "MPLBACKEND": "Agg",
}
# The KF notebook override (identical text to kf/kernel notebook).
FIXES_ENV = {"LOCAL_ANALYZER_YIELD_SECONDS": "300", "DUCK_PROMPT_GAMEOVER_RETRY": "1", "DUCK_PROMPT_LEVELUP_REVIEW": "1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=sorted(BUNDLES))
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("games", nargs="+")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    key = json.loads((Path.home() / ".local/share/opencode/auth.json").read_text())["openrouter"]["key"]
    os.environ.update(ANALYZER_ENV)
    os.environ["LOCAL_ANALYZER_API_KEY"] = key
    os.environ["OPENAI_API_KEY"] = key
    if args.arm == "kf":
        os.environ.update(FIXES_ENV)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RECORDINGS_DIR"] = str(out_dir / "server_recording")

    bundle = BUNDLES[args.arm]
    sys.meta_path[:] = [finder for finder in sys.meta_path if "__editable__" not in repr(finder)]
    for repo in sorted((bundle / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))

    import arc_agi
    import taaf.game_api
    import inference.agent.tool_agent as tool_agent

    assert Path(tool_agent.__file__).is_relative_to(bundle), tool_agent.__file__
    print(f"SMOKE arm={args.arm} tool_agent={tool_agent.__file__} yield={tool_agent._LOCAL_ANALYZER_YIELD_SECONDS}"
          f" retry={getattr(tool_agent, '_PROMPT_GAMEOVER_RETRY', None)} review={getattr(tool_agent, '_PROMPT_LEVELUP_REVIEW', None)}",
          flush=True)

    with open(bundle / "deploy_target.pkl", "rb") as file:
        target = pickle.load(file)
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    with open(bundle / "benchmark_initial.pkl", "rb") as file:
        bm = pickle.load(file)
    bm.job_dir = out_dir
    bm.solver.max_runtime_s_per_game = args.minutes * 60.0
    bm.solver.analyzer_timeout = 900.0
    bm.solver.concurrency = args.concurrency
    bm.solver.max_actions_per_game = None
    bm.solver.save_request_logs = True

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    by_prefix = {env.game_id.split("-")[0]: env.game_id for env in arcade.available_environments}
    bm.games = [taaf.game_api.GameAPI(env_name=by_prefix[g], arcade_spec=spec) for g in args.games]
    bm.n_passes = 1
    bm.game_weights = None

    soft_end = datetime.now() + timedelta(minutes=args.minutes + 10)
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True))
    bm._save_json()
    for run in bm.game_runs:
        print(f"SMOKE_RESULT {run.game_id} state={run.state} score={run.final_score} actions={len(run.history)}", flush=True)


if __name__ == "__main__":
    main()
