"""One gameplay arm per fresh interpreter: no shared scheduling clock."""
import ast
import asyncio
from datetime import datetime, timedelta
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time
from urllib.request import urlopen


async def main(config_path):
    cfg=json.loads(Path(config_path).read_text())
    sources=json.loads(Path(cfg['phase_sources']).read_text())
    ns=dict(globals(),BUNDLE_DIR=Path(cfg['bundle_dir']),WORKING_DIR=Path(cfg['working_dir']),
            TRUE_SUBMISSION=False,USE_PRIORITY_SCHEDULING=True,PLAY_S=cfg['play_s'],NOTEBOOK_START_EPOCH=time.time())
    for i in ['14','16','20']:
        result=eval(compile(sources[i],f'isolated-phase-{i}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),ns)
        if inspect.isawaitable(result):await result
        if i=='14':
            from inference.agent import tool_agent
            gate=tool_agent._PRIORITY_GATE
            receipt={'pid':os.getpid(),'parent_pid':os.getppid(),'gate_is_fresh':gate is None,'play_s':cfg['play_s']}
            (Path(cfg['working_dir'])/'harness-isolation.json').write_text(json.dumps(receipt,indent=2)+'\n')
            if gate is not None:raise RuntimeError('harness process already has a scheduling gate')


if __name__=='__main__':asyncio.run(main(sys.argv[1]))
