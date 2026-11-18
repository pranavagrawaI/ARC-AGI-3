"""Compare preserved scientific code, normalize only declared reversed-order branch/estimand."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def check():
 unchanged={}
 for name in ('api_arm.py','accounting.py','load_contract.py','catalog.py','catalog.json','corpus_base.py','candidate_library.py','candidate_resources.py'):
  old=(BASE/'trial-16-candidate-s10-r4-operational-local'/name).read_bytes();new=(HERE/name).read_bytes()
  if old!=new:raise ValueError('undeclared scientific source change '+name)
  unchanged[name]=hashlib.sha256(new).hexdigest()
 old=(BASE/'trial-15-wire-capacity-local/reduce_api.py').read_text();new=(HERE/'reduce_api.py').read_text()
 before="if a['capture']['end']>b['capture']['start'] or (a['capture']['server_pgid'],a['capture']['server_start'])==(b['capture']['server_pgid'],b['capture']['server_start']):issues.append('fresh sequential loads')"
 after="try:\n            import handoff\n            handoff.pair(b['capture'],a['capture'])\n        except (ValueError,KeyError,TypeError):issues.append('fresh candidate-first sequential loads')"
 if old.count(before)!=1 or new.count(after)!=1:raise ValueError('exact reverse-order delta missing')
 old=old.replace(before,'pass').replace('WIRE_EQUIVALENT_TOTAL_PACKAGE_API_CAPACITY','WIRE_EQUIVALENT_FIXED_CANDIDATE_FIRST_TOTAL_PACKAGE_API_CAPACITY');new=new.replace(after,'pass')
 def compare(source):return next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='compare')
 if ast.dump(compare(old),include_attributes=False)!=ast.dump(compare(new),include_attributes=False):raise ValueError('paired scientific gate drift beyond declared order/estimand')
 return dict(unchanged=unchanged,pair_science_AST_equal_after_exact_declared_order_estimand_delta=True,historical_rate_gate=False,threshold=1.20,baseline65_rerun=False)
if __name__=='__main__':print(json.dumps(check(),indent=2))
