import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from solver_agent import MdArcSolverAgent
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

def response(code, identity):
    return LLMResponse(raw_response=None, content='',
        tool_calls=[ToolCall(id=identity, name='execute_python', arguments=json.dumps({'code':code}))],
        finish_reason='tool_calls', assistant_message={'role':'assistant','content':''})

async def main():
    fake = FakeLLMClient(scripted_responses=[
        response("self.submit_actions(['NOT_LEGAL'], rationale='invalid attempt')", 'rejected'),
        response("self.submit_actions(['UP'], rationale='repaired')", 'accepted'),
    ])
    agent = MdArcSolverAgent(llm=fake, run_dir=Path.cwd())
    while not (Path('ipc/states.jsonl').stat().st_size):
        await asyncio.sleep(.01)
    await agent.handle({'user_messages':['Make one legal move.']})
asyncio.run(main())
