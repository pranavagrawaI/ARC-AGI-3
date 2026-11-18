"""Bounded read-only source/receipt audit; writes only this audit's derived output."""
import argparse,ast,hashlib,json,math,re,signal,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REPO=BASE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600);started=time.monotonic()
 trial=BASE/'trial-7-serving-feasibility-guard';native=trial/'remote-readback/archive-extracted/serving-candidate-feas7';data=native/'output'
 plan=json.loads((trial/'sources/plan.json').read_text());identity=json.loads((data/'installed-source-identity.json').read_text())
 overlay=REPO/'analysis/m2-winners-20261001/audit/lord-runtime/overlay'
 modules=['vllm/models/qwen4_exp/nvidia/ngram_embedding.py','vllm/models/qwen4_exp/nvidia/model.py','vllm/model_executor/model_loader/default_loader.py','vllm/model_executor/model_loader/weight_utils.py','vllm/v1/worker/gpu_model_runner.py']
 for n in modules:assert sha(overlay/n)==identity[n],n
 config=BASE/'trial-3-matched-preparation/processor-assets/candidate/config.json';assert sha(config)==plan['model']['assets']['config.json']['sha256']
 c=json.loads(config.read_text())['text_config'];launch=json.loads((data/'candidate-launch.json').read_text());argv=launch['argv']
 assert argv[argv.index('--tensor-parallel-size')+1]=='1' and argv[argv.index('--quantization')+1]=='compressed-tensors'
 assert json.loads(argv[argv.index('--engram-config')+1])=={'cpu_offload':True} and argv[argv.index('--cpu-offload-params')+1]=='embed_tokens'
 assert '--safetensors-load-strategy' not in argv
 source=(overlay/modules[0]).read_text();tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Qwen4ExpNGramEmbedding')
 names={'_is_prime_64','_nth_prime_after','_make_vocab_layout'};methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names];assert len(methods)==3
 wrapper=ast.ClassDef(name='Layout',bases=[],keywords=[],body=methods,decorator_list=[]);module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]));ns={};exec(compile(module,'<pinned-pure-layout>','exec'),ns)
 assert c['ple_layer_ids']==[2];heads=(c['ngram_size']-1)*c['heads_per_ngram'];sizes,_,total=ns['Layout']._make_vocab_layout(ngram_vocab_size_base=c['ngram_vocab_size_base'],ngram_heads=heads,ple_dense_layer_id=0)
 # Independent trial-division crosscheck of the16 prime vocabulary sizes.
 independent=[];candidate=c['ngram_vocab_size_base']
 while len(independent)<heads:
  if candidate>=2 and all(candidate%d for d in range(2,math.isqrt(candidate)+1)):independent.append(candidate)
  candidate+=1
 assert independent==sizes
 divisor=c['make_ngram_vocab_size_divisible_by'];padded=(total+divisor-1)//divisor*divisor;ple=padded*(c['ple_embed_dim']//heads)*2;embedding=c['vocab_size']*c['hidden_size']*2
 assert ple==102400491520 and embedding==1271398400
 files=plan['model']['files'];weights=[r for r in files if r['name'].endswith('.safetensors')];ple_files=[r for r in files if r['name'].startswith('ple-bf16-')]
 log=(data/'candidate.server.stdout').read_text();lines=log.splitlines()
 wanted=[dict(line=i+1,text=s) for i,s in enumerate(lines) if any(t in s for t in ('input embedding kept in pinned','Initialized PLE embedding','Filesystem type for checkpoints','Skipping auto-prefetch','Loading weights took','Model loading took','Actual usage is'))]
 assert 'weight_dtype=torch.bfloat16, weight_device=cpu, pinned=True' in log
 trace=[json.loads(s) for s in (data/'supervisor.jsonl').read_text().splitlines()];startup=[r for r in trace if r['event']=='resource']
 assert [r['name'] for r in startup]==['meminfo','cgroup_memory_max','cgroup_cpu_max']
 guard=json.loads((data/'host-headroom.json').read_text());capacity=guard['effective_memory_bytes'];assert capacity==175*1024**3
 assert 'memory.current' not in (trial/'sources/prepare.py').read_text()
 trial12=BASE/'trial-12-bootstrap-local';series=json.loads((trial12/'remote-readback/derived-original/SAMPLE-SERIES.json').read_text());joined=[r for r in series if r['nonhost_join']];assert len(joined)==82
 last=series[-1];assert all(not r['eligible']['cgroup'] for r in series)
 refs=[config,trial/'sources/plan.json',trial/'sources/prepare.py',trial/'sources/candidate_library.py',data/'installed-source-identity.json',data/'candidate-launch.json',data/'candidate.server.stdout',data/'host-headroom.json',data/'supervisor.jsonl',native/'supervisor.py',data/'result.json',BASE/'trial-8-wire-capacity-local/release_check.py',BASE/'trial-8-wire-capacity-local/EXPERIMENT-CARD.md',BASE/'trial-3-matched-preparation/EXPERIMENT-CARD.md',BASE/'trial-4-saturated-local/EXPERIMENT-CARD.md',BASE/'shutdown-release-prototype-local-20261023/HOST-THRESHOLD-TRACE.md',trial12/'resources.py',trial12/'remote-readback/derived-original/SAMPLE-SERIES.json',trial12/'remote-readback/derived-original/COLLECTION-PROOF.json']+[overlay/n for n in modules]
 inputs={str(p):sha(p) for p in refs}
 report=dict(status='LOCAL_READ_ONLY_HOST_ADMISSION_AUDIT_NOT_A_NEW_GATE',seconds=time.monotonic()-started,inputs=inputs,exact_installed_source_pins_verified=modules,model_config_sha256=sha(config),
  provenance=dict(earliest_inspected_contract='trial8 release_check.py:21 and card:49; retained unchanged through release12',independent_peak_derivation_found=False,scope='retained trial3/4/7/8/10/12 artifacts and prior source trace; not exhaustive outside workspace',startup_guard='min(total host capacity,cgroup memory.max)>=170GiB, not remaining current/free/peak'),
  candidate=dict(model=plan['model']['model_version'],runtime=plan['runtime_dataset'],tensor_parallel=1,PLE_class='Qwen4ExpPLEUnquantizedEmbeddingMethod despite FP8 metadata, because compressed-tensors branch',PLE_dtype='BF16',PLE_device='pinned CPU',heads=heads,head_dim=c['ple_embed_dim']//heads,total_vocab_rows=total,padded_vocab_rows=padded,PLE_tensor_bytes=ple,input_embedding_bytes=embedding,known_pinned_tensor_bytes=ple+embedding,known_pinned_tensor_GiB=(ple+embedding)/1024**3,
   checkpoint_weight_bytes=sum(r['size'] for r in weights),checkpoint_weights_GiB=sum(r['size'] for r in weights)/1024**3,checkpoint_shards=len(weights),PLE_file_bytes=sum(r['size'] for r in ple_files),PLE_file_headers_and_metadata_excess_over_tensor_bytes=sum(r['size'] for r in ple_files)-ple,
   peak_incremental_cgroup_charge_bytes=None,permanent_tensor_is_not_peak=True,source='PinnedHostEmbedding torch.empty(cpu,pin_memory=True), independent shape/prime derivation; no tensors allocated by audit'),
  trial7=dict(total_capacity_bytes=capacity,startup_cgroup_current_captured=False,continuous_host_series_captured=False,cgroup_peak_captured=False,functional_result=json.loads((data/'result.json').read_text()),runtime_log_facts=wanted,
   available_log_semantics='pinned weight_utils min(psutil host available,cgroup limit minus get_cgroup_memory_usage()); callee implementation not retained in this audit, no inactive-file semantic assumed',
   observed_available_RAM_GiB=[10.78,3.42],available_snapshots_after_construction_not_entry_peak=True,model_loading_71_91_GiB_is_GPU_context_not_CPU_peak=True,
   default_safetensors_branch='None: auto-prefetch only if NFS and checkpoint<=90percent available; actual log skips it, safe_open.get_tensor iteration; no whole-file eager f.read path selected',
   image_grid_scope='6 full-image diagnostic29-token print(1) replies, not S10 full-response workload fit'),
  trial12=dict(samples=len(series),all_nonhost_joined_samples=len(joined),host_eligible_samples=0,host_threshold_bytes=140*1024**3,host_max_bytes=max(r['host_bytes'] for r in series),last_observation=last['host_input'],last_headroom_bytes=last['host_bytes'],last_headroom_GiB=last['host_bytes']/1024**3,
   gap_above_known_pinned_tensor_bytes=last['host_bytes']-(ple+embedding),gap_is_not_proven_margin=True,active_file_is_not_reclaimability_proof=True),
  decisions=dict(original_140_gate='NOT QUALIFIED; unchanged; no data-backed new threshold from this audit',source_bound_candidate_requirement='At least96.551971GiB for named pinned tensors, plus UNKNOWN other resident/temporary charges; cannot admit at that lower bound',
   smallest_discriminating_measurement='Separately judged one postcontrol candidate load/health-only feasibility with phase-bound memory.current/stat/oom events and process pinned/RSS peaks, normal loader behavior, no cache intervention/replay. Measures incremental demand in the relevant state; no packet/authority prepared here.',
   alternatives=['Retain140 and stop: valid existing contract but current paired run remains blocked','Derive conservative incremental upper bound+independent margin and verify capacity against it; no such bound exists yet','If judgment permits bounded measurement risk, measure exact candidate after control once; admission conditions must be declared separately before any run, not fitted to111.48','Changing PLE format/runtime or host envelope changes tested package/design and requires separate qualification']),
  limitations=['File/active/inactive counters do not prove reclaimability or attribute charges to control files','A fresh-session fit and low loader-time AvailableRAM do not establish fit from postcontrol state','Checkpoint bytes, GPU weight footprint and host incremental peak are different quantities','No source/controller/threshold/cache/packet change or GPU/network/API/model work'])
 (out/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
 for p,h in inputs.items():assert sha(p)==h,p
 print(json.dumps({k:report[k] for k in ('status','seconds','candidate','trial12')},indent=2));signal.alarm(0)
if __name__=='__main__':main()
