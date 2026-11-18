"""Five final resource probes using accepted controller-owned query containment."""
import json
from pathlib import Path
import time
import adapters
import capture
import candidate_scope
from candidate_evidence import host_formula
from core import resources


class Adapters(adapters.Adapters):
    def __init__(self, root, binding, uuid, paths, port, end, scope,
                 commands=None, execution_policy='FUTURE_REVIEWED_GPU_QUERY'):
        self.root = Path(root)
        self.binding, self.uuid, self.paths = binding, uuid, paths
        self.port, self.end, self.scope = port, end, scope
        self.commands = commands or {'gpu': capture.GPU_ARGV, 'apps': capture.APPS_ARGV}
        self.execution_policy = execution_policy
        self.owner = capture.owner_identity()
        self.captured = {}
        self.check_scope()

    def check_scope(self):
        return candidate_scope.check(self.scope, self.binding, self.paths)

    def cgroup(self):
        before = self.check_scope()
        raw, refs, stable = {}, {}, True
        for name in ('meminfo', 'maximum', 'current', 'stat'):
            data, refs[name], same = adapters.snapshot(self.paths[name], self.root, name, self.end)
            stable = stable and same
            if len(data) > 65536:
                raise ValueError('host parser bound')
            raw[name] = data.decode().strip() if name in ('maximum', 'current') else data.decode()
        after = self.check_scope()
        value = host_formula(raw)
        value.pop('raw')
        return dict(**value, artifacts=refs, files_stable_during_read=stable,
            input_paths={name: str(self.paths[name]) for name in raw},
            input_scope='CANDIDATE_PROCESS_CGROUP_NO_NATIVE_PROXY',
            scope_before=before, scope_after=after,
            observation='sequential non-atomic reads; original140 report only')

    def native(self):
        raise ValueError('native proxy is forbidden in candidate-only screen')

    def probes(self):
        adapter = self
        class Prepared(dict):
            def prepare_controller(self, end):
                return adapter.prepare_controller(end)
        return Prepared(gpu=lambda: self.query('gpu', capture.GPU_ARGV),
            apps=lambda: self.query('apps', capture.APPS_ARGV), cgroup=self.cgroup,
            port=self.port_probe, disk=self.disk)


def accepted(sample, deadline):
    """Original140 is measured and checked for shape, never required true."""
    try:
        probes = sample['probes']
        if (set(probes) != {'gpu', 'apps', 'cgroup', 'port', 'disk'}
                or sample['probe_worker_alive'] or sample['ipc_error'] is not None
                or not sample['process']['eligible'] or sample['process']['remaining']
                or not sample['start'] <= sample['end'] <= deadline):
            return False
        for name, value in probes.items():
            if 'error' in value or not sample['start'] <= value['start'] <= value['end'] <= sample['end']:
                return False
            if name != 'cgroup' and value['eligible'] is not True:
                return False
        host = probes['cgroup']
        return (type(host['eligible']) is bool and type(host['computed_bytes']) is int
            and host['computed_bytes'] >= 0 and host['threshold_bytes'] == 140 * 1024**3
            and host['files_stable_during_read'] is True)
    except (KeyError, TypeError, ValueError):
        return False
