"""CPU-only feasibility probe. No model calls, gameplay, uploads, or incumbent edits."""
import ast,base64,collections,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'solvers/m2_b1/src'))
from inference.agent import vision_context as V
from inference.utils.animation import normalize_frames,collapse_frames,_transient_cells
os.environ.update(MULTIMODAL_CONTEXT='current_grid',MULTIMODAL_UPSCALE='10',ARC3_ANIMATION='1')
started=time.monotonic()
track_path=ROOT/'analysis/rule-learning-prefix-20261017/inputs.json'
replay_path=ROOT/'analysis/failure-audit-20261013/tn36/engine-replay.json'
track=next(t for t in json.loads(track_path.read_text()) if t['id']=='tn36-b1-r1')
replay=json.loads(replay_path.read_text())
command=next(c for c in replay['runs']['b1-r1']['commands'] if c['local_action']==41)
before=normalize_frames([track['actions'][39]['board']])[0]
chain=collapse_frames(before,normalize_frames(command['animation_frames']))
cells=_transient_cells(chain)
interior_changed=sum(a!=b for row1,row2 in zip(before[4:-4],chain[-1][4:-4]) for a,b in zip(row1[4:-4],row2[4:-4]))
assert interior_changed==0 and cells
part,index=V.animation_peak_part(chain,cells)
assert part is not None
# Verify the rendering preserves the selected raw frame; no oracle annotation enters it.
assert chain[index+1] in normalize_frames(command['animation_frames'][i] for i in replay['missed_rotation_probe']['rotated_shape_visible_in_animation_frames'])
(OUT/'tn36-peak.png').write_bytes(base64.b64decode(part['image_url']['url'].split(',',1)[1]))
for name,grid in [('before',before),('settled',chain[-1])]:
 data=V.frame_to_png_data_url(SimpleNamespace(grid=grid))
 (OUT/f'tn36-{name}.png').write_bytes(base64.b64decode(data.split(',',1)[1]))
# Execute the existing method body with actual rendering helpers, isolated from model/network imports.
source_path=ROOT/'solvers/m2_b1/src/inference/agent/tool_agent.py'
tree=ast.parse(source_path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_prepare_animation_image')
namespace={'Any':object,'_animation_enabled':lambda:True,'animation_image_mode':V.animation_image_mode,'_animation_transient_cells':_transient_cells,'animation_peak_part':V.animation_peak_part,'animation_composite_part':V.animation_composite_part,'_get_env_int':lambda name,default:default}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source_path),'exec'),namespace)
def exercise(mode,changed=False,transient_count=len(cells)):
 os.environ['ARC3_ANIMATION_IMAGE']=mode
 holder=SimpleNamespace(_last_action_gameplay_changed=changed,_last_animation_chain=chain,_pending_animation_image=None,_pending_animation_image_label='')
 namespace['_prepare_animation_image'](holder,{'animation':{'transient_pixels':transient_count}})
 return holder
assert exercise('off')._pending_animation_image is None
positive=exercise('peak');assert positive._pending_animation_image==part
assert exercise('peak',changed=True)._pending_animation_image is None
assert exercise('peak',transient_count=0)._pending_animation_image is None
cohort=ROOT/'solvers/m2_f1x_reap/experiment/screen-1/merge/output'
counts=collections.Counter();by_game={};source_hashes={}
for p in sorted(cohort.glob('*_requests.jsonl')):
 seen=set();local=collections.Counter();h=hashlib.sha256()
 for line in p.open('rb'):
  h.update(line);r=json.loads(line)
  if r.get('event')!='request':continue
  local['requests']+=1
  users=[m for m in r.get('messages',[]) if m.get('role')=='user']
  content=users[-1]['content'] if users else ''
  text=content if isinstance(content,str) else '\n'.join(x.get('text','') for x in content if isinstance(x,dict) and x.get('type')=='text')
  if 'Animation image (frame ' in text or 'Animation image: the cells' in text:local['automatic_image_request_exposures']+=1
  eligible='left the board area unchanged, but it was not a no-op:' in text
  if not eligible:continue
  local['candidate_request_exposures']+=1
  fingerprint=hashlib.sha256(text.encode()).hexdigest()
  if fingerprint not in seen:local['distinct_candidate_user_messages']+=1;seen.add(fingerprint)
 counts.update(local);by_game[p.name.split('-')[0]]=dict(local);source_hashes[str(p.relative_to(ROOT))]=h.hexdigest()
for p in [track_path,replay_path,source_path,Path(V.__file__),ROOT/'solvers/m2_b1/src/inference/utils/animation.py']:
 source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
result={'question':'Can an existing automatic real-frame rendering option expose a real transient cue, and is that exposure absent from D16?','status':'CPU_RENDERING_FEASIBLE_MODEL_BENEFIT_UNTESTED','case':{'game':'tn36','run':'b1-r1','local_action':41,'chain_frames':len(chain),'transient_pixels':len(cells),'interior_changed_pixels_at_settle':interior_changed,'selected_engine_frame_index':index,'selected_frame_contains_archived_visible_rotation':True,'before_shape':list(map(len,[before,before[0]]))},'controls':{'mode_off_emits_no_image':True,'peak_mode_emits_exact_real_frame':True,'changed_board_emits_no_image':True,'zero_transient_pixels_emits_no_image':True},'d16_scan':dict(counts),'by_game':by_game,'source_sha256':source_hashes,'elapsed_s':time.monotonic()-started,'limits':['One selected cue, not prevalence of consequential missing observations.','Prompt eligibility is counted from exact existing text, not a full ToolAgent execution.','Isolated production method and actual renderer exercised; no whole-runtime/package qualification.','Images were already available on request; improvement from automatic delivery is untested.','Automatic images may distract or increase input/prefill cost.','Peak is one real frame; it can omit evidence visible at other moments.']}
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['by_game','source_sha256']},indent=2))
