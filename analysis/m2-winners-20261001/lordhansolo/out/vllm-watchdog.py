'''Keep the vLLM OpenAI server alive for the whole harness run.

Started by the notebook setup step with the path of a JSON config holding the server command and
the server log, pid and ready-marker paths. The server is spawned as a child and awaited with a
blocking wait, so a dead server is noticed without polling. Every exit is logged with its code,
uptime and the tail of the server log. The server is restarted after a short pause, which lets the
engine process that the dying API server kills release the GPU and the port. It is not restarted
when it died before the setup step marked it ready, so a broken launch fails the setup step, and
not more than MAX_RESTARTS times, so a crash that comes back deterministically ends in a readable
"giving up" line instead of a restart loop for the rest of the run.

Each server start runs the GPU shard prefetcher from that start's log offset, so a restart never
mistakes the previous load's completed progress bar for its own. Prefetch reports are kept
separately for the initial start and every restart. The prefetch runs in a separate process,
with bounded termination waits so a blocked NFS read cannot hold up server restarts.
'''
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def is_test_run() -> bool:
    '''Return true only for an explicit noncompetition test run.'''

    return (
        os.environ.get('TAAF_RUN_AS_SUBMISSION', '').strip().lower() == '0'
        and os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'', '0', 'false'}
    )


MAX_RESTARTS = 5
RESTART_DELAY_SECONDS = 10
HOST_MEMORY_SAMPLE_INTERVAL_SECONDS = 15
GPU_SHARD_PREFETCH_THREADS = 4
GPU_SHARD_PREFETCH_WINDOW_BYTES = 8 * 1024**3
GPU_SHARD_PREFETCH_BLOCK_BYTES = 16 * 1024**2
GPU_SHARD_PREFETCH_POLL_SECONDS = 0.5
GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS = 2
PLE_SHARD_NAME_PREFIX = 'ple-'
GPU_SHARD_PREFETCH_ABORT = threading.Event()
WATCHDOG_EXIT_DEFERRED = False
WATCHDOG_EXIT_REQUESTED = False

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
server_command = config['command']
server_log = Path(config['server_log'])
server_pid = Path(config['server_pid'])
server_ready = Path(config['server_ready'])


def read_integer_file(path: str) -> int | None:
    '''Read a cgroup counter, returning None for unreadable or nonnumeric values.'''
    try:
        return int(Path(path).read_text(encoding='ascii').strip())
    except (OSError, ValueError):
        return None


def read_host_memory() -> dict[str, int | None]:
    '''Report available RAM, including the cgroup limit used by vLLM's loader.'''
    host_available_bytes = None
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        if line.startswith('MemAvailable:'):
            host_available_bytes = int(line.split()[1]) * 1024
            break
    if host_available_bytes is None:
        raise ValueError('MemAvailable is missing from /proc/meminfo')

    cgroup_limit_bytes = read_integer_file('/sys/fs/cgroup/memory.max')
    if cgroup_limit_bytes is not None:
        cgroup_usage_bytes = read_integer_file('/sys/fs/cgroup/memory.current')
    else:
        cgroup_limit_bytes = read_integer_file('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if cgroup_limit_bytes is not None and cgroup_limit_bytes >= 1 << 62:
            cgroup_limit_bytes = None
        cgroup_usage_bytes = (
            read_integer_file('/sys/fs/cgroup/memory/memory.usage_in_bytes')
            if cgroup_limit_bytes is not None else None
        )

    cgroup_available_bytes = None
    if cgroup_limit_bytes is not None:
        cgroup_available_bytes = (
            cgroup_limit_bytes
            if cgroup_usage_bytes is None else max(0, cgroup_limit_bytes - cgroup_usage_bytes)
        )
    available_bytes = (
        min(host_available_bytes, cgroup_available_bytes)
        if cgroup_available_bytes is not None else host_available_bytes
    )

    return {
        'available_bytes': available_bytes,
        'host_available_bytes': host_available_bytes,
        'cgroup_available_bytes': cgroup_available_bytes,
        'cgroup_limit_bytes': cgroup_limit_bytes,
        'cgroup_usage_bytes': cgroup_usage_bytes,
    }


def log_host_memory() -> None:
    '''Sample host RAM throughout server startup, inference, and restarts.'''
    output_path = Path(config['host_memory_log'])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as output:
            while True:
                record = {'ts': round(time.time(), 3), **read_host_memory()}
                output.write(json.dumps(record) + '\n')
                output.flush()
                time.sleep(HOST_MEMORY_SAMPLE_INTERVAL_SECONDS)
    except (OSError, ValueError) as error:
        log_event(f'host memory monitor stopped: {error!r}')


def log_event(message: str) -> None:
    print(f'{datetime.now().isoformat(timespec="seconds")} watchdog: {message}', flush=True)


def read_server_log_tail(lines: int = 40) -> str:
    if not server_log.exists():
        return ''

    return '\n'.join(server_log.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def build_natural_sort_key(path: Path) -> list:
    '''Split a file name into text and integer parts, the key vLLM's safetensors loader sorts by.'''
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path.name)]


def list_loader_shards(model_path: Path) -> list[Path]:
    '''Return the checkpoint shards of the index in the order vLLM's safetensors loader reads them.'''
    index = json.loads((model_path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    shards = [model_path / name for name in set(index['weight_map'].values())]

    return sorted(shards, key=build_natural_sort_key)


def read_file_into_page_cache(path: Path, stop_event: threading.Event) -> dict:
    '''Read a file in large blocks so its pages sit in the page cache, until the end or a stop.

    The read stops early when its stop_event or GPU_SHARD_PREFETCH_ABORT is set. Both are checked
    between blocks, so a stop takes effect only after an in-flight read returns. Returns the
    monotonic start and end of the read and whether it was stopped early.
    '''
    started_at = time.monotonic()
    buffer = memoryview(bytearray(GPU_SHARD_PREFETCH_BLOCK_BYTES))
    with path.open('rb', buffering=0) as handle:
        while not (stopped := stop_event.is_set() or GPU_SHARD_PREFETCH_ABORT.is_set()) and handle.readinto(buffer):
            pass

    return {'started_at': started_at, 'finished_at': time.monotonic(), 'stopped': stopped}


def drop_file_from_page_cache(path: Path) -> None:
    '''Ask the kernel to evict a file's clean cached pages. The kernel treats this as a hint.

    An eviction that fails only leaves the pages in the cache, so it is reported and the prefetch
    carries on with a window that is smaller in practice than it is on paper.
    '''
    try:
        file_descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(file_descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        print(f'GPU shard prefetch could not evict {path.name}: {error!r}', flush=True)


class LoaderProgressFollower:
    '''Follow the completed-shard count of the target load's progress bar in the vLLM server log.

    The bar prints `| N/<shard_count> [` after the N-th shard, one update per line. The MTP draft's
    second bar has a different total, so it never matches. The unterminated tail of each read is
    kept and completed by the next one, so a line cut by a read is still counted. The count is -1
    until the bar's first line, which the loader prints once the pinned PLE table is allocated.
    '''

    def __init__(self, log_handle, shard_count: int) -> None:
        self._log_handle = log_handle
        self._pattern = re.compile(rf'\| (\d+)/{shard_count} \[')
        self._unterminated_text = ''
        self.loaded_shard_count = -1

    def update_loaded_shard_count(self) -> int:
        '''Read the new log text and return the highest completed count seen so far.'''
        complete_text, _, self._unterminated_text = (self._unterminated_text + self._log_handle.read()).rpartition('\n')
        for match in self._pattern.finditer(complete_text):
            self.loaded_shard_count = max(self.loaded_shard_count, int(match.group(1)))

        return self.loaded_shard_count


class GpuShardPrefetcher:
    '''Keep the GPU shards a bounded window ahead of vLLM's single-stream loader in the page cache.

    The loader reads the GPU shards through one NFS stream at 44-153 MiB/s, while the PLE shards
    already arrive at 615-923 MiB/s through its multi-threaded copy, so only the GPU shards are
    prefetched. vLLM's own prefetch (v264, v265) read ahead with no bound, and after the pinned PLE
    table the page cache holds about 20 GiB, so pages far ahead were evicted before the loader got
    there. Nothing is read before the loader prints its first progress line, so the window never
    competes with the PLE allocation. From then on shards are submitted in load order while the
    bytes submitted and not yet passed by the loader fit the window, and the shard being loaded is
    always submitted.

    A shard the loader has passed gets its read stopped, and once that read has ended its pages
    are evicted. Evicting while the read still runs would let it refill the cache behind the loader.
    Pages read twice (the prefetch, then the loader) sit on the active LRU list and would
    otherwise push the window's unread pages out first.

    The loop returns once every GPU shard is passed and evicted, and writes a JSON report with the
    per-shard prefetch and loader times, relative to the loader's first progress line. If the bar
    never appears, nothing is read and no report is written. Shards the loader crossed inside one
    poll carry the timestamp of the next printed update, so complete_before_loader_reached is an
    upper bound and the report counts them. The log offset isolates each server start's progress.
    '''

    def __init__(self, shards: list[Path], server_log: Path, report_path: Path, log_offset: int = 0) -> None:
        self._shards = shards
        self._server_log = server_log
        self._report_path = report_path
        self._log_offset = log_offset
        self._gpu_shard_indices = [
            index for index, shard in enumerate(shards) if not shard.name.startswith(PLE_SHARD_NAME_PREFIX)
        ]
        self._shard_sizes = {index: shards[index].stat().st_size for index in self._gpu_shard_indices}
        self._futures: dict[int, concurrent.futures.Future] = {}
        self._stop_events: dict[int, threading.Event] = {}
        self._progress_reached_at: dict[int, float] = {}
        self._submitted_position = 0
        self._passed_position = 0
        self._dropped_position = 0

    def run(self) -> None:
        '''Follow the loader until every GPU shard is passed and evicted, then write the report.

        GPU_SHARD_PREFETCH_ABORT ends the loop as well as the reads, and a pass that ends early
        writes no report.
        '''
        while not self._server_log.exists():
            if GPU_SHARD_PREFETCH_ABORT.is_set():
                return
            time.sleep(GPU_SHARD_PREFETCH_POLL_SECONDS)
        with (
            self._server_log.open(encoding='utf-8', errors='replace') as log_handle,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=GPU_SHARD_PREFETCH_THREADS, thread_name_prefix='gpu-shard-prefetch'
            ) as executor,
        ):
            log_handle.seek(self._log_offset)
            follower = LoaderProgressFollower(log_handle, len(self._shards))
            while self._dropped_position < len(self._gpu_shard_indices) and not GPU_SHARD_PREFETCH_ABORT.is_set():
                loaded_shard_count = follower.update_loaded_shard_count()
                if loaded_shard_count >= 0:
                    self._record_progress(loaded_shard_count)
                    self._stop_passed_shards(loaded_shard_count)
                    self._drop_passed_shards()
                    self._submit_window(executor)
                time.sleep(GPU_SHARD_PREFETCH_POLL_SECONDS)
        if self._dropped_position == len(self._gpu_shard_indices) and self._progress_reached_at:
            self._write_report()

    def _record_progress(self, loaded_shard_count: int) -> None:
        '''Stamp the first time each completed count was seen, including counts a poll skipped.

        tqdm drops the updates that fall inside its minimum interval, so counts it never printed
        carry the timestamp of the next printed one, which is later than the real crossing. Those
        shards show equal reached and passed times in the report.
        '''
        now = time.monotonic()
        for count in range(len(self._progress_reached_at), loaded_shard_count + 1):
            self._progress_reached_at[count] = now

    def _stop_passed_shards(self, loaded_shard_count: int) -> None:
        '''Stop the reads of GPU shards the loader has finished and cancel those still queued.'''
        while (
            self._passed_position < len(self._gpu_shard_indices)
            and self._gpu_shard_indices[self._passed_position] < loaded_shard_count
        ):
            shard_index = self._gpu_shard_indices[self._passed_position]
            if shard_index in self._futures:
                self._stop_events[shard_index].set()
                self._futures[shard_index].cancel()
            self._passed_position += 1

    def _drop_passed_shards(self) -> None:
        '''Evict passed GPU shards in order, each only after its read has ended.'''
        while self._dropped_position < self._passed_position:
            shard_index = self._gpu_shard_indices[self._dropped_position]
            if shard_index in self._futures and not self._futures[shard_index].done():
                return
            drop_file_from_page_cache(self._shards[shard_index])
            self._dropped_position += 1

    def _submit_window(self, executor: concurrent.futures.ThreadPoolExecutor) -> None:
        '''Submit the next GPU shards while the bytes still held in the page cache fit the window.

        The window counts from the first shard not yet evicted, not from the loader, so shards the
        loader has passed while their read winds down keep their bytes in the budget.
        '''
        self._submitted_position = max(self._submitted_position, self._passed_position)
        ahead_bytes = sum(
            self._shard_sizes[index]
            for index in self._gpu_shard_indices[self._dropped_position:self._submitted_position]
        )
        while self._submitted_position < len(self._gpu_shard_indices) and (
            self._submitted_position == self._passed_position
            or ahead_bytes + self._shard_sizes[self._gpu_shard_indices[self._submitted_position]]
            <= GPU_SHARD_PREFETCH_WINDOW_BYTES
        ):
            shard_index = self._gpu_shard_indices[self._submitted_position]
            self._stop_events[shard_index] = threading.Event()
            self._futures[shard_index] = executor.submit(
                read_file_into_page_cache, self._shards[shard_index], self._stop_events[shard_index]
            )
            ahead_bytes += self._shard_sizes[shard_index]
            self._submitted_position += 1

    def _describe_shard(self, shard_index: int, loader_started_at: float) -> dict:
        '''Return one shard's report row, with times in seconds after the loader's first progress line.'''
        row = {
            'name': self._shards[shard_index].name,
            'gib': round(self._shard_sizes[shard_index] / 1024**3, 3),
            'loader_reached': round(self._progress_reached_at[shard_index] - loader_started_at, 1),
            'loader_passed': round(self._progress_reached_at[shard_index + 1] - loader_started_at, 1),
        }
        future = self._futures.get(shard_index)
        if future is None or future.cancelled():
            row['prefetch'] = 'not started'
        elif future.exception() is not None:
            row['prefetch'] = f'failed: {future.exception()!r}'
        else:
            read = future.result()
            row['prefetch'] = 'stopped' if read['stopped'] else 'complete'
            row['prefetch_started'] = round(read['started_at'] - loader_started_at, 1)
            row['prefetch_finished'] = round(read['finished_at'] - loader_started_at, 1)

        return row

    def _write_report(self) -> None:
        '''Write the window settings and one row per GPU shard to the report file.'''
        loader_started_at = self._progress_reached_at[0]
        rows = [self._describe_shard(index, loader_started_at) for index in self._gpu_shard_indices]
        report = {
            'threads': GPU_SHARD_PREFETCH_THREADS,
            'window_gib': GPU_SHARD_PREFETCH_WINDOW_BYTES / 1024**3,
            'gpu_shards': len(self._gpu_shard_indices),
            'gpu_shard_gib': round(sum(self._shard_sizes.values()) / 1024**3, 2),
            'other_shards': len(self._shards) - len(self._gpu_shard_indices),
            'complete_before_loader_reached': sum(
                row['prefetch'] == 'complete' and row['prefetch_finished'] <= row['loader_reached'] for row in rows
            ),
            'shards_crossed_within_one_poll': sum(row['loader_reached'] == row['loader_passed'] for row in rows),
            'shards': rows,
        }
        self._report_path.write_text(json.dumps(report, indent=1), encoding='utf-8')


def prefetch_gpu_shards(log_offset: int, report_path: Path) -> None:
    '''Build the GPU shard prefetcher for the mounted model and run it.

    It runs in a separate process, so a failure while reading the checkpoint index
    only ends the prefetch and leaves the server launch alone.
    '''
    GpuShardPrefetcher(
        list_loader_shards(Path(config['model_path'])), server_log, report_path, log_offset
    ).run()


def start_gpu_shard_prefetch(log_offset: int, report_path: Path) -> subprocess.Popen | None:
    '''Launch an isolated prefetch process, leaving the server running if spawning fails.'''
    try:

        return subprocess.Popen([
            sys.executable, __file__, sys.argv[1], '--prefetch', str(log_offset), str(report_path)
        ])
    except OSError as error:
        log_event(f'could not start GPU shard prefetch: {error!r}')

        return None


def stop_gpu_shard_prefetch(prefetch_process: subprocess.Popen) -> None:
    '''Terminate prefetch without waiting indefinitely for a blocked filesystem read.'''
    if prefetch_process.poll() is not None:

        return
    prefetch_process.terminate()
    try:
        prefetch_process.wait(timeout=GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS)

        return
    except subprocess.TimeoutExpired:
        log_event(f'GPU shard prefetch did not stop within {GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS}s; sending SIGKILL')
    prefetch_process.kill()
    try:
        prefetch_process.wait(timeout=GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log_event('GPU shard prefetch is still blocked after SIGKILL; continuing without waiting for it')


@contextmanager
def defer_watchdog_exit():
    '''Finish registering or stopping prefetch before honoring a watchdog exit request.'''
    global WATCHDOG_EXIT_DEFERRED
    WATCHDOG_EXIT_DEFERRED = True
    try:
        yield
    finally:
        WATCHDOG_EXIT_DEFERRED = False
        if WATCHDOG_EXIT_REQUESTED:
            raise SystemExit(0)


def exit_watchdog(signal_number: int, frame) -> None:
    '''Request shutdown once, deferring it while prefetch ownership changes or cleanup runs.'''
    global WATCHDOG_EXIT_REQUESTED
    if WATCHDOG_EXIT_REQUESTED:

        return
    WATCHDOG_EXIT_REQUESTED = True
    if not WATCHDOG_EXIT_DEFERRED:
        raise SystemExit(0)


def run_watchdog() -> None:
    '''Spawn the server, wait for it to exit, log the exit and start it again while that is allowed.'''
    restart_count = 0
    prefetch_process = None
    while True:
        with server_log.open('a' if restart_count else 'w', encoding='utf-8') as server_log_handle:
            if restart_count:
                server_log_handle.write(
                    f'\n==== watchdog restart {restart_count} at {datetime.now().isoformat(timespec="seconds")} ====\n'
                )
                server_log_handle.flush()
            log_offset = server_log_handle.tell()
            started_at = time.monotonic()
            server_process = subprocess.Popen(
                server_command, stdout=server_log_handle, stderr=subprocess.STDOUT, text=True
            )
            server_pid.write_text(str(server_process.pid), encoding='utf-8')
            log_event(f'vLLM server started (pid {server_process.pid}, restart {restart_count})')
            report_path = Path(config['shard_prefetch_report'])
            if restart_count:
                report_path = report_path.with_name(f'{report_path.stem}-restart-{restart_count}{report_path.suffix}')
            try:
                with defer_watchdog_exit():
                    if prefetch_process is None or prefetch_process.poll() is not None:
                        prefetch_process = start_gpu_shard_prefetch(log_offset, report_path)
                    else:
                        log_event('previous GPU shard prefetch is still exiting; skipping another prefetch')
                exit_code = server_process.wait()
                log_event(f'vLLM server exited with code {exit_code} after {time.monotonic() - started_at:.0f}s')
            finally:
                with defer_watchdog_exit():
                    if prefetch_process is not None:
                        stop_gpu_shard_prefetch(prefetch_process)
        log_event('last server log lines:\n' + read_server_log_tail())
        if not server_ready.exists():
            log_event('server died before it was ready; not restarting')
            sys.exit(1)
        if restart_count >= MAX_RESTARTS:
            log_event(f'giving up after {MAX_RESTARTS} restarts; the server stays down')
            sys.exit(1)
        restart_count += 1
        log_event(f'restarting vLLM server in {RESTART_DELAY_SECONDS}s (restart {restart_count} of {MAX_RESTARTS})')
        time.sleep(RESTART_DELAY_SECONDS)


if len(sys.argv) > 2 and sys.argv[2] == '--prefetch':
    prefetch_gpu_shards(int(sys.argv[3]), Path(sys.argv[4]))
    sys.exit(0)
signal.signal(signal.SIGTERM, exit_watchdog)
if is_test_run():
    threading.Thread(target=log_host_memory, name='host-memory-monitor', daemon=True).start()
run_watchdog()
