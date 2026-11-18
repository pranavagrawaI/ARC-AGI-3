"""Bounded read-only live-log snapshot of the frozen kernel identity."""
import datetime
import json
import signal
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionLogsStreamRequest

HERE = Path(__file__).resolve().parent
lines = []


def stop(*_):
    raise TimeoutError('bounded log snapshot')


signal.signal(signal.SIGALRM, stop)
signal.alarm(25)
try:
    api = KaggleApi()
    api.authenticate()
    identity = json.loads((HERE / 'identity.json').read_text())
    user, slug = identity['metadata']['id'].split('/')
    query = ApiGetKernelRequest()
    query.user_name, query.kernel_slug = user, slug
    with api.build_kaggle_client() as client:
        kernel = client.kernels.kernels_api_client.get_kernel(query)
    assert kernel.metadata.id == 138477950 and kernel.metadata.current_version_number == 1
    query = ApiGetKernelSessionLogsStreamRequest()
    query.user_name, query.kernel_slug = user, slug
    query.wait_for_logs_url_seconds = 1
    with api.build_kaggle_client() as client:
        with client.kernels.kernels_api_client.get_kernel_session_logs_stream(query) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
                if line:
                    lines.append(line if isinstance(line, str) else line.decode(errors='replace'))
except Exception as exc:
    print(type(exc).__name__, str(exc)[:200])
finally:
    signal.alarm(0)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
path = HERE / 'evidence' / f'live-log-{stamp}.txt'
path.write_text('\n'.join(lines))
decoded = []
for line in lines:
    if line.startswith('data: '):
        try:
            row = json.loads(line[6:])
            decoded.append(f"{row.get('time', '')}: {row.get('data', '')}")
        except (ValueError, TypeError):
            pass
print(path)
print(''.join(decoded)[-5000:])
