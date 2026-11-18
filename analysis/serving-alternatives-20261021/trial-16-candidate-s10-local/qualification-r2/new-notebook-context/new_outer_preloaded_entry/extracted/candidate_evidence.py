"""Independent candidate-only raw resources, actual identity and deadline checks."""
import hashlib
import json
import math
from pathlib import Path
import time
import artifacts
import resources
import capture as query_capture


def require(value, message):
    if not value:
        raise ValueError('candidate evidence: ' + message)


def read(path):
    return json.loads(Path(path).read_text())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def raw(root, ref):
    require(artifacts.verify(root / 'query-artifacts', ref, time.monotonic()+60), 'incomplete raw artifact')
    return (root / 'query-artifacts' / ref['path']).read_bytes()


def host_formula(raw):
    require(set(raw)=={'meminfo','maximum','current','stat'},'complete host schema')
    require(raw['maximum']=='max' or raw['maximum'].isdigit(),'numeric or unlimited cgroup max')
    require(raw['current'].isdigit(),'numeric current cgroup')
    fields=dict(line.split() for line in raw['stat'].splitlines())
    require('inactive_file' in fields and all(value.isdigit() for value in fields.values()),'complete nonnegative cgroup stat')
    available=[line.split(':',1)[1].split() for line in raw['meminfo'].splitlines() if line.startswith('MemAvailable:')]
    require(len(available)==1 and len(available[0])==2 and available[0][0].isdigit() and available[0][1]=='kB','measured MemAvailable schema')
    return resources.cgroup(raw)


def verify_sample(root, sample, binding, uuid, deadline, allow_cpu=False):
    require(all(number(sample[k]) for k in ('start', 'end', 'ipc_deadline'))
        and sample['start'] <= sample['end'] <= deadline and sample['ipc_deadline'] <= deadline,
        'resource observation deadline')
    require(not sample['probe_worker_alive'] and sample['ipc_error'] is None
        and sample['process']['eligible'] is True and sample['process']['remaining'] == [],
        'resource/process evidence incomplete')
    probes = sample['probes']
    require(set(probes) == {'gpu', 'apps', 'cgroup', 'port', 'disk'}, 'five candidate probes; no native proxy')
    for record in probes.values():
        require('error' not in record and all(number(record[k]) for k in ('start','end'))
            and sample['start'] <= record['start'] <= record['end'] <= sample['end'], 'resource receipt clocks/error')
    controller = sample['controller_queries']
    require(controller['all_queries_cleaned_before_worker'] is True
        and controller['owner_pid'] == binding['controller']['pid'], 'query owner binding')
    for name in ('gpu', 'apps'):
        command = probes[name]['command']
        require(command == controller['commands'][name] and command['ok'] is True
            and command['cleanup_verified'] is True and command['containment']['verified'] is True,
            'query integrity/containment/cleanup')
        intended = query_capture.GPU_ARGV if name == 'gpu' else query_capture.APPS_ARGV
        require(command['intended_argv'] == intended and command['owner_pid'] == binding['controller']['pid']
            and command['child_reaped'] is True and command['retained_query'] is None
            and command['error'] is None and command['cleanup_error'] is None
            and sample['start'] <= command['start'] <= command['execution_authorized_at']
                <= command['cleanup_end'] <= command['end'] <= sample['ipc_deadline'], 'query raw ownership/clocks')
        if not allow_cpu:
            require(command['execution_scope'] == 'FUTURE_REVIEWED_GPU_QUERY'
                and Path(command['executed_argv'][0]).name == 'nvidia-smi', 'actual GPU query command')
        for ref in list(command['artifacts'].values()) + [command['intent'],command['containment']['artifact']]:
            raw(root, ref)
        value = raw(root, command['artifacts']['stdout']).decode()
        if name == 'gpu':
            expected = resources.gpu(value, uuid)
            require(all(probes[name][key] == expected[key] for key in ('eligible','used_mib','threshold_mib')),
                    'raw GPU rederivation')
        else:
            require(probes[name]['eligible'] == (not value.strip()), 'raw apps rederivation')
    process=sample['process']
    require(sample['start']<=process['start']<=process['end']<=sample['end'],'process observation clocks')
    host = probes['cgroup']
    inputs = {key: raw(root, ref).decode() for key, ref in host['artifacts'].items()}
    require(set(inputs) == {'meminfo','maximum','current','stat'}, 'complete raw host evidence')
    inputs.update({key: inputs[key].strip() for key in ('maximum','current')})
    expected = host_formula(inputs)
    require(host['files_stable_during_read'] is True
        and all(host[key] == expected[key] for key in ('eligible','computed_bytes','threshold_bytes')),
        'original140 raw formula/stability drift')
    scope=read(root/'candidate-scope.json')
    require(host['input_paths']=={key:scope['paths'][key] for key in ('meminfo','maximum','current','stat')}
        and host['input_scope']=='CANDIDATE_PROCESS_CGROUP_NO_NATIVE_PROXY','raw host input scope')
    for observation in (host['scope_before'],host['scope_after']):
        require(observation['cgroup']==scope['observer']['cgroup'] and observation['namespaces']==scope['observer']['namespaces']
            and observation['run_id']==scope['run_id'] and observation['input_scope']==host['input_scope'],'host scope before/after')
    disk = json.loads(raw(root, probes['disk']['artifact']))
    require(disk['path']==scope['paths']['disk']
        and probes['disk']['start']<=disk['start']<=disk['end']<=probes['disk']['end'],'disk raw scope/clocks')
    require(probes['disk']['free_bytes'] == disk['free_bytes']
        and probes['disk']['threshold_bytes'] == resources.DISK_MIN_BYTES
        and probes['disk']['eligible'] == (disk['free_bytes'] >= resources.DISK_MIN_BYTES),
        'raw disk rederivation')
    port = json.loads(raw(root, probes['port']['artifact']))
    require(port['reuseaddr'] is True and probes['port']['start']<=port['start']<=port['end']<=probes['port']['end'],'port raw clocks/reuse')
    require(port['address'] == ['127.0.0.1',8001] and type(port['eligible']) is bool
        and probes['port']['eligible'] == port['eligible']
        and (port['error'] is None) == port['eligible'], 'raw free port')
    return dict(original140=expected['eligible'], original140_bytes=expected['computed_bytes'],
        eligible=all(probes[name]['eligible'] for name in ('gpu','apps','disk','port')))


def final_resources(root, config, events, capture):
    receipt = read(root / 'candidate-release.json')
    binding = read(root / 'candidate-scope.json')['binding']
    root_identity = binding['members'][str(binding['pgid'])]
    require(str(root_identity['pgid']) == capture['server_pgid']
        and str(root_identity['starttime']) == capture['server_start']
        and root_identity['boot_id'] == capture['boot_id'], 'actual replay/server cleanup identity')
    starts = [row for row in events if row['event'] == 'candidate_cleanup_start']
    require(len(starts) == 1, 'one candidate cleanup start')
    begin = starts[0]
    expected_end = min(events[0]['work_deadline'], begin['cleanup_start'] + config['final_resource_seconds'])
    require(receipt['absolute_deadline'] == begin['phase_deadline'] == expected_end
        and receipt['server_pgid'] == binding['pgid'] == begin['server_pgid']
        and receipt['start'] == begin['cleanup_start'], 'configured final180/work deadline')
    cleaned = [row for row in events if row['event'] == 'cleanup' and row['pgid'] == binding['pgid']]
    require(len(cleaned) == 1 and cleaned[0]['remaining'] == []
        and begin['t'] <= cleaned[0]['t'] <= expected_end, 'actual whole-group cleanup late/missing')
    require(receipt['pass_'] is True and receipt['error'] is None
        and receipt['start'] <= receipt['end'] <= expected_end, 'resource phase failure/lateness')
    require(capture['end'] <= begin['cleanup_start'], 'cleanup preceded replay completion')
    observed = []
    accepted = []
    previous=cleaned[0]['t']
    for index,sample in enumerate(receipt['samples']):
        require(previous <= sample['start'] <= sample['end'] <= receipt['end'], 'post-cleanup sample clocks')
        value = verify_sample(root, sample, binding, capture['gpu_uuid'], expected_end,
                              allow_cpu=config['mode']=='CPU_FIXTURE_ONLY')
        observed.append(value)
        previous=sample['end']
        if value['eligible']:
            accepted.append(index)
    require(bool(observed) and accepted == [len(observed)-1], 'first eligible final resource sample missing')
    passes=[row for row in events if row['event']=='candidate_resources_pass']
    require(len(passes)==1 and all(passes[0][key]==receipt[key] for key in ('start','end','absolute_deadline'))
        and receipt['end']<=passes[0]['t']<=expected_end,'resource pass trace join')
    return dict(pass_=True, original140=observed[-1]['original140'], host_report_only=True,
        native_proxy=False, actual_server_pgid=binding['pgid'], absolute_deadline=expected_end)


def package(root, config, capture):
    plan = read(Path(__file__).with_name('candidate-plan.json'))
    folder = root / 'candidate-package'
    source = read(folder / 'installed-source-identity.json')
    require(all(source[key] == value for key,value in plan['candidate_frontend_source_sha'].items()), 'candidate installed source drift')
    for name,expected in plan['candidate_frontend_source_sha'].items():
        require(hashlib.sha256((folder/'installed-source'/name).read_bytes()).hexdigest()==expected,'actual installed candidate source bytes '+name)
    model = read(folder / 'model-source-identity.json')
    require(model['version'] == plan['model']['model_version']
        and all(model['asset_sha256'][key] == value['sha256'] for key,value in plan['model']['assets'].items()),
        'candidate model/processor identity')
    require(model['files']=={row['name']:int(row['size']) for row in plan['model']['files']}
        and model['full_weight_sha256_verified'] is False,'complete pinned model filename/size inventory; no full weight hash claim')
    prepared=read(folder/'prepared.json')
    require(prepared==dict(manifest_sha=plan['runtime_manifest_sha'],runtime_root='/tmp/serving-candidate-s10-16-runtime',rate_comparison=False),'prepared runtime manifest/root')
    reference=read(Path(__file__).with_name('candidate-runtime-reference.json'))
    versions=read(folder/'installed-package-identity.json')
    require(all(versions[key]==reference['versions'][key] for key in ('vllm','torch','transformers','cuda_build'))
        and versions['python'].startswith('3.12.'),'installed runtime versions; patch gap reported')
    launch = read(folder / 'candidate-launch.json')
    require(launch==reference['launch'],'exact candidate argv/environment flags')
    require(launch['image_max_pixels'] == 409600 and launch['image_limit'] == 58
        and launch['sequences'] == 10 and launch['native_overlay_enabled'] is False, 'candidate flags')
    scope = read(root / 'candidate-scope.json')
    identity = scope['binding']['members'][str(scope['binding']['pgid'])]
    expected_argv = hashlib.sha256(b''.join(s.encode()+b'\0' for s in launch['argv'])).hexdigest()
    require(identity['cmdline_sha256'] == expected_argv and str(identity['pgid']) == capture['server_pgid']
        and str(identity['starttime']) == capture['server_start'] and identity['boot_id'] == capture['boot_id']
        and scope['run_id'] == capture['session_id'] == config['run_id'], 'actual argv/session/starttime binding')
    return True


def host_evidence(root,config,events):
    """Rederive every required host/OOM row; sampled/namespace limits remain."""
    scope=read(root/'observer-scope.json')
    candidate_scope=read(root/'candidate-scope.json')
    require(scope['observer_scope']==candidate_scope['observer']
        and scope['run_id']==candidate_scope['run_id']==config['run_id'],'observer/candidate scope join')
    require(all(scope['paths'][key]==candidate_scope['paths'][key]
        and scope['file_keys'][key]==candidate_scope['file_keys'][key]
        for key in candidate_scope['paths']), 'candidate/observer input path/file join')
    boundaries=[json.loads(row) for row in (root/'host-boundaries.jsonl').read_bytes().splitlines()]
    series_bytes=(root/'host-series.jsonl').read_bytes()
    rows=[json.loads(row) for row in series_bytes.splitlines()]
    report=read(root/'host-series.result.json')
    require(report['pass_'] is True and report['error'] is None and rows and boundaries
        and report['samples']==len(rows) and report['bytes']==len(series_bytes)
        and report['series_sha256']==hashlib.sha256(series_bytes).hexdigest(),'host series completeness')
    recorded=[event for event in events if event['event']=='host_boundary']
    require(len(recorded)==len(boundaries) and all((row['label'],row['start'],row['end'],row['valid'],row['invalidity'])
        ==(event['label'],event['start'],event['end'],event['valid'],event['invalidity'])
        for row,event in zip(boundaries,recorded)), 'boundary trace/raw join')
    require(boundaries[0]['label']=='before_candidate_preparation' and boundaries[-1]['label']=='final_cleanup', 'host lifecycle boundaries')
    values=[]
    for group in (boundaries,rows):
        previous=-math.inf
        for row in group:
            require(all(number(row[key]) for key in ('start','end')) and previous<=row['start']<=row['end']<=events[0]['work_deadline'], 'host clocks/work end')
            require(row['run_id']==config['run_id'] and row['scope']==scope['observer_scope']
                and row['paths']==scope['paths'], 'host observation scope drift')
            require(set(row.get('invalidity',[]))<={'OOM counter increased; stop load test'}, 'late/malformed host boundary')
            raw_events={key:int(value) for key,value in (line.split() for line in row['raw']['events'].splitlines())}
            require(raw_events==row['oom_events'] and all(type(raw_events[key]) is int and raw_events[key]>=0 for key in ('oom','oom_kill')), 'raw OOM consistency')
            host_raw={key:row['raw'][key].strip() if key in ('maximum','current') else row['raw'][key] for key in ('meminfo','maximum','current','stat')}
            derived=host_formula(host_raw)
            if 'original140' in row:require(row['original140']==derived, 'original140 boundary rederivation')
            for key in scope['required']:
                clock=row['file_clocks'][key]
                require(row['start']<=clock['start']<=clock['end']<=row['end'], 'host raw read clock')
            values.append((row['start'],tuple(raw_events[key] for key in ('oom','oom_kill'))))
            previous=row['end']
    baseline=tuple(boundaries[0]['oom_events'][key] for key in ('oom','oom_kill'))
    # Overlapping sequential reads are not atomically ordered; counters must never fall below baseline.
    require(all(all(value>=base for value,base in zip(counts,baseline)) for _,counts in values),'OOM reset/inconsistent baseline')
    increased=any(counts!=baseline for _,counts in values)
    require(rows[0]['start']<=boundaries[1]['start'] and rows[-1]['end']>=boundaries[-1]['end'], 'observer must cover workload/final boundary')
    first_increase=min((clock for clock,counts in values if counts!=baseline),default=None)
    return dict(first_increase=first_increase,oom_increased=increased,boundaries=len(boundaries),samples=len(rows),original140=boundaries[-1]['original140']['eligible'])


def check(root,config,primary_error=None,failed_stage=None):
    import candidate_reduce,outcomes
    root=Path(root)
    events=[json.loads(line) for line in (root/'supervisor.jsonl').read_bytes().splitlines()]
    require(events and [row['seq'] for row in events]==list(range(len(events)))
        and events[0]['event']=='start' and events[0]['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(), 'supervisor sequence/config')
    hardware=[row for row in events if row['event']=='hardware_pass']
    require(len(hardware)==1 and hardware[0]['name']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
        and hardware[0]['memory_mib']>=90000 and hardware[0]['driver'], 'actual hardware')
    scope=read(root/'candidate-scope.json');identity=scope['binding']['members'][str(scope['binding']['pgid'])]
    root_retained=[row for row in events if row['event']=='candidate_root_retained']
    require(len(root_retained)==1 and all(root_retained[0]['identity'][key]==identity[key]
        for key in ('pid','starttime','boot_id','pgid','sid','uid')), 'retained initial root lifetime')
    spawns=[row for row in events if row['event']=='spawn' and row['phase']=='candidate.server']
    require(len(spawns)==1 and spawns[0]['pid']==identity['pid'] and spawns[0]['argv']==config['candidate']['server']['argv'], 'one actual candidate server spawn')
    require(not any(row['event']=='spawn' and row['phase'].startswith('control') for row in events), 'no prior control workload')
    value=candidate_reduce.arm(root)
    fallback=dict(server_pgid=str(identity['pgid']),server_start=str(identity['starttime']),
        boot_id=identity['boot_id'],session_id=config['run_id'],gpu_uuid=hardware[0]['uuid'],end=root_retained[0]['t'])
    captured=value.get('capture',fallback)
    raw_events=root/'candidate/events.jsonl'
    if raw_events.exists():
        import load_contract
        replay_events=[json.loads(line) for line in raw_events.read_bytes().splitlines()]
        boot=load_contract.validate_log(replay_events,'replay_start')
        header=replay_events[0]
        captured={key:header[key] for key in ('session_id','gpu_uuid','server_pgid','server_start')}
        captured.update(boot_id=boot,origin=header['t'],start=header['t'],end=replay_events[-1]['t'])
        starts=[row for row in events if row['event']=='spawn' and row['phase']=='candidate.replay']
        exits=[row for row in events if row['event']=='exit' and row['phase']=='candidate.replay']
        bound=[row for row in events if row['event']=='candidate_server_bound']
        require(len(starts)==len(exits)==len(bound)==1
            and bound[0]['identity']==identity and bound[0]['session_id']==config['run_id']
            and bound[0]['gpu_uuid']==hardware[0]['uuid'] and bound[0]['t']<=starts[0]['t']<=captured['origin']<=captured['end']<=exits[0]['t'], 'actual health-bound replay phase clocks')
        require(any(row['event']=='ready' and row['arm']=='candidate' and row['t']<=bound[0]['t'] for row in events),'actual candidate health before replay')
    require(captured['gpu_uuid']==hardware[0]['uuid'] and captured['boot_id']==events[0]['boot_id'], 'same actual hardware/session')
    package(root,config,captured)
    resource=final_resources(root,config,events,captured)
    host=host_evidence(root,config,events)
    failure=None
    if host['oom_increased']:
        prepare=[row for row in events if row['event']=='host_boundary' and row['label']=='candidate.prepare.before']
        failure=dict(kind='OBSERVED_OOM' if prepare and host['first_increase']>=prepare[0]['end'] else 'PRE_CANDIDATE_OR_AMBIGUOUS_OOM',raw_verified=bool(prepare) and host['first_increase']>=prepare[0]['end'])
    elif primary_error is not None:
        failure=dict(kind='UNKNOWN_OR_HARNESS_FAILURE',raw_verified=False)
        if failed_stage=='startup':
            cleanups=[row for row in events if row['event']=='candidate_cleanup_start']
            health=[row for row in events if row['event']=='exit' and row.get('phase','').startswith('candidate.health-')]
            if cleanups[0]['cleanup_start']>=root_retained[0]['startup_deadline'] and health and all(row['returncode']==75 for row in health):
                failure=dict(kind='STARTUP_NOT_READY_AT_CAP',raw_verified=True)
    elif not value['valid']:
        # These four issues are returned only after complete raw protocol/accounting validation.
        complete={'genuine corpus starvation','client generator not saturated','fewer than 30 completions','block instability'}
        failure=dict(kind='COMPLETE_WITHIN_ARM_GATE_FAILURE' if set(value['issues'])<=complete else 'UNVERIFIED_ARM_OR_HTTP_FAILURE',
                     raw_verified=bool(value['issues']) and set(value['issues'])<=complete)
    if primary_error is not None and failed_stage=='replay' and not host['oom_increased']:
        candidate_failure=candidate_reduce.complete_api_failure(root)
        if candidate_failure is not None:failure=candidate_failure
    if primary_error is None:
        labels=['before_candidate_preparation','corpus.prepare.before','corpus.prepare.after','candidate.prepare.before',
            'candidate.prepare.after','candidate.identity.before','candidate.identity.after','candidate.server.spawned',
            'candidate.health200','candidate.replay.before','candidate.replay.after','candidate.qualify.after','candidate.cleaned','final_cleanup']
        require([row['label'] for row in map(json.loads,(root/'host-boundaries.jsonl').read_text().splitlines())]==labels,'complete successful phase boundary coverage')
    interpretation=outcomes.classify(dict(integrity_complete=True,identity_bound=True,cleanup_complete=True,
        within_arm_valid=value['valid'],U_resp=value.get('U_resp'),failure=failure))
    return dict(**interpretation,resource_evidence=resource,host_evidence=host,within_arm=value,
        failure_evidence=failure,limitations=['sampled, cumulative cgroup peak; shared RSS; no peak safety margin',
        'version/weight index-name-size and asset checks; no full checkpoint rehash',
        'source-origin guard is not live-code or hostile-hook attestation',
        'inherited64/65 conservative proc identity abort and delayed precutoff EOF limit'])
