"""Actual repaired q6 CodeAct dispatcher against a paused captured history."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import resource

# Bound generated in-process cells before numerical/framework imports.
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))

HERE = Path(__file__).resolve().parent
from probe import KICKOFF, rows

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--offline-smoke', action='store_true')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    (run_dir/'RESOURCE-LIMITS.json').write_text(json.dumps({'address_space_bytes':resource.getrlimit(resource.RLIMIT_AS),'cpu_seconds':resource.getrlimit(resource.RLIMIT_CPU)})+'\n')
    candidate = Path(os.environ['Q6_REPLAY_CANDIDATE'])
    sys.path[:0] = [str(candidate),str(candidate/'chassis')]
    spec = importlib.util.spec_from_file_location('replay_headless',candidate/'headless.py')
    headless = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(headless)
    llm = None
    if args.offline_smoke:
        from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
        code = "self.submit_actions(['UP'], rationale='predict: move', predicted_path=[{'expect':'move'}])"
        llm = FakeLLMClient(scripted_responses=[LLMResponse(raw_response=None,content='',tool_calls=[ToolCall(id='smoke',name='execute_python',arguments=json.dumps({'code':code}))],finish_reason='tool_calls',assistant_message={'role':'assistant','content':''})])
    else:
        llm = headless.build_llm()
    agent = headless.MdArcSolverAgent(llm=llm,
        truncation=headless.TruncationConfig(max_context_tokens=32000,
            capture=headless.CaptureConfig(max_stdout=6000),
            event_format=headless.FormatConfig(max_string=2000,max_length=100,max_depth=4)),
        run_dir=run_dir, game_id='the game', alias='the game', reflect_every=8,
        visual='off',png_scale=8,skill_path=candidate/'chassis/skills/interactive-game-solver/SKILL.md',max_actions_per_turn=20)
    if (candidate/'chassis/prediction_feedback.py').exists():
        from prediction_feedback import PredictionFeedback
        from solver_agent import _check_helper_ast
        feedback = PredictionFeedback(agent.ipc_dir,agent.helpers_dir,_check_helper_ast)
        states = rows(agent.ipc_dir/'states.jsonl')
        previous = next(s for s in states if s['step']==5)
        feedback.commit(feedback.prepare(previous,['LEFT']))
        feedback.observe()
    agent._user_messages_in.put(KICKOFF)
    dispatch = asyncio.create_task(headless._headless_dispatch(agent))
    async def submitted():
        while not (agent.ipc_dir/'actions.jsonl').stat().st_size:
            if (agent.ipc_dir/'ABORT').exists():
                raise RuntimeError((agent.ipc_dir/'ABORT').read_text())
            await asyncio.sleep(.05)
        (run_dir/'REPLAY-COMPLETE.json').write_text(json.dumps({'accepted_submission':True,'executed_actions':0})+'\n')
    monitor = asyncio.create_task(submitted())
    try:
        done,_ = await asyncio.wait([dispatch,monitor],return_when=asyncio.FIRST_COMPLETED)
        for task in done: task.result()
    finally:
        dispatch.cancel(); monitor.cancel()
        await asyncio.gather(dispatch,monitor,return_exceptions=True)
        await agent.queue_manager.shutdown()

if __name__=='__main__':
    asyncio.run(main())
