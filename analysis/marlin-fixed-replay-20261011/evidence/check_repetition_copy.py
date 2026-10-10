import sys,json,tempfile,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));import benchmark
class Tensor:
 def __init__(self,data):self.data=np.array(data)
 def clone(self):return Tensor(self.data.copy())
 def copy_(self,other):self.data[:]=other.data if isinstance(other,Tensor) else other
 def __mul__(self,value):return Tensor(self.data*value)
 def __add__(self,value):return Tensor(self.data+value)
 def __mod__(self,value):return Tensor(self.data%value)
class Graph:
 def __init__(self):self.output=Tensor(np.ones((2,4),np.float32));self.calls=0
 def replay(self):self.output.data[:]=1+self.calls*.00001;self.calls+=1
graphs=[Graph() for _ in range(4)]
args=[{'hidden_states':Tensor(np.ones((2,4))),'topk_ids':Tensor(np.zeros((2,1),np.int32)),'topk_weights':Tensor(np.ones((2,1)))} for _ in graphs]
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None))
 with patch.dict(sys.modules,{'torch':fake}),patch.object(benchmark,'cpu_array',side_effect=lambda t:t.data),patch.object(benchmark,'fp32_reference',return_value=Tensor(np.ones((2,4),np.float32))),patch.object(benchmark,'dtype_reference',return_value=Tensor(np.ones((2,4),np.float32))):
  benchmark.check_graph_outputs(graphs,[g.output for g in graphs],args,{'w1':np.zeros((4,1))},Tensor(np.ones((2,4),np.float32)),p)
 d=np.load(p/'original-repeats.npz')
 for name in ['baseline','treatment','independent_control']:
  expected=np.array([1+i*.00001 for i in range(1,9)],np.float32)
  np.testing.assert_array_equal(d[name][:,0,0],expected)
 print(json.dumps({'passed':True,'contrast':'actual graph-output repetition caller with reusable overwritten buffers','snapshots_preserved':24,'variants':3}))
