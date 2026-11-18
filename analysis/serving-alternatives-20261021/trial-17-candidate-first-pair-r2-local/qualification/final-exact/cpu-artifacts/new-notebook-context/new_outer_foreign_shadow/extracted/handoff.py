"""Reverse-order clocks shared by producer and independently reconstructed consumer."""
import math

def require(ok,message):
 if not ok:raise ValueError('reverse handoff: '+message)

def clocks(release,fresh,admitted,dispatch,work_end,seconds):
 values=[release[k] for k in ('start','end','absolute_deadline')]+[fresh[k] for k in ('start','end','ipc_deadline')]+[admitted,dispatch,work_end,seconds]
 require(all(type(x) in (int,float) and math.isfinite(x) for x in values),'finite clocks')
 deadline=min(work_end,release['start']+seconds)
 require(release['pass_'] is True and release['error'] is None and release['absolute_deadline']==deadline,'original shared180/work deadline')
 require(release['start']<=release['end']<=fresh['start']<=fresh['end']<=admitted<=dispatch<=deadline,'first/fresh/dispatch chronology')
 require(fresh['end']<=fresh['ipc_deadline']<=deadline and dispatch-fresh['end']<=2,'sole fresh/actual dispatch <=2s')
 return deadline

def pair(candidate,control):
 for key in ('session_id','gpu_uuid','boot_id'):
  require(candidate[key]==control[key],'same-run '+key)
 require(candidate['start']<=candidate['end']<=control['start']<=control['end'],'candidate first chronology')
 require((candidate['server_pgid'],candidate['server_start'])!=(control['server_pgid'],control['server_start']),'fresh second-control identity')
 return True
