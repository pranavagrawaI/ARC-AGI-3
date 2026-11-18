import json
import os
from pathlib import Path
import time
import candidate_prepare
import replay

if __name__=='__main__':
    started=time.monotonic()
    mount=candidate_prepare.dataset('pranavagrawai','mtp-depth-replay-data-20261006')
    index=replay.prepare(mount,Path(__file__).parent/'wire-bodies')
    Path(os.environ['SCREEN_OUTPUT'],'corpus-preparation.json').write_text(json.dumps(dict(records=len(index),mount=str(mount),elapsed_seconds=time.monotonic()-started))+'\n')
