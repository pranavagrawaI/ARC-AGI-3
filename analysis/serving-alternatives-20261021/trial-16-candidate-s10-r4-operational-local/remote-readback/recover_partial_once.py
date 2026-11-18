"""Reuse the prior strict recovery implementation; no duplicate recoverer/API."""
import hashlib, importlib.util, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
TOOL=HERE.parents[1]/'trial-15-wire-capacity-local/remote-readback/recover_final_partial.py'
SPEC=importlib.util.spec_from_file_location('prior_strict_partial_recoverer',TOOL)
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)
module.ARCHIVE=HERE/'archive-v1/output.zip'
module.EXPECTED='de1677dd95cd77656ece739e071ba99fb89e475a1c65f5eba5cb2f05cb6fe550'
destination=HERE/'partial-recovery-v1'
with (HERE/'partial-recovery-invocation.json').open('x') as stream:
    json.dump(dict(tool=str(TOOL),tool_sha256=hashlib.sha256(TOOL.read_bytes()).hexdigest(),
        original_tool_unmodified=True,archive=str(module.ARCHIVE),archive_sha256=module.EXPECTED,
        destination=str(destination),CPU_seconds=30,member_bytes=32*1024**2,decoded_bytes=512*1024**2,
        remote_calls=0,scope='sequential header+descriptor+CRC; stop first incomplete; no resynchronization'),stream,indent=2)
module.recover(destination)
assert hashlib.sha256(module.ARCHIVE.read_bytes()).hexdigest()==module.EXPECTED
