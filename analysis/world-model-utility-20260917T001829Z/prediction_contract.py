"""Strict typed observable contract. No semantic transition rule."""
import copy

def grid(value,shape=None):
 if not isinstance(value,list) or not value or not all(isinstance(r,list) and r for r in value):raise ValueError('grid must be complete nonempty integer rows')
 dims=(len(value),len(value[0]))
 if any(len(r)!=dims[1] or any(type(x) is not int for x in r) for r in value):raise ValueError('grid must be rectangular with integer cells (no bool/float/nonfinite)')
 if shape is not None and dims!=tuple(shape):raise ValueError('grid dimension mismatch')
 return dims

def commitment(value,shape):
 if not isinstance(value,dict):raise ValueError('commitments must be a typed object')
 if value.get('type')=='grid' and set(value)=={'type','predicted_grid'}:
  grid(value['predicted_grid'],shape);return copy.deepcopy(value)
 if value.get('type')=='abstain' and set(value)=={'type','reason'}:
  if not isinstance(value['reason'],str) or not value['reason'].strip():raise ValueError('abstain requires nonblank reason')
  return copy.deepcopy(value)
 raise ValueError('commitments must be exactly type=grid plus predicted_grid OR type=abstain plus reason')

def branch(value,shape):
 if not isinstance(value,list) or len(value)!=1:raise ValueError('predict must return exactly one branch')
 item=value[0]
 if not isinstance(item,dict) or not {'state','commitments'}<=set(item) or set(item)-{'state','commitments','declared_dependencies'}:raise ValueError('branch keys: state, commitments, optional declared_dependencies only')
 deps=item.get('declared_dependencies',[])
 if not isinstance(deps,list) or not all(isinstance(x,str) for x in deps):raise ValueError('declared_dependencies must be string array')
 return commitment(item['commitments'],shape)

def score(current,target,prediction):
 shape=grid(current);grid(target,shape);c=commitment(prediction,shape)
 C=[x for row in current for x in row];Y=[x for row in target for x in row];n=len(C)
 D=[i for i in range(n) if C[i]!=Y[i]];U=[i for i in range(n) if C[i]==Y[i]]
 base={'exact_match':C==Y,'cell_accuracy':len(U)/n,'changed_cell_accuracy':0.0 if D else None,'false_changes':0,'false_change_rate':0.0 if U else None}
 out={'coverage':int(c['type']=='grid'),'abstained':c['type']=='abstain','cells':n,'changed_cells':len(D),'unchanged_cells':len(U),'baseline':base,'prediction':None,'paired_cell_accuracy_difference':None,'paired_exact_difference':None,'predictive_success':False,'inference':'descriptive one-transition only'}
 if c['type']=='abstain':out['abstention_reason']=c['reason'];return out
 P=[x for row in c['predicted_grid'] for x in row];accuracy=sum(x==y for x,y in zip(P,Y))/n;false=sum(P[i]!=C[i] for i in U)
 out.update(prediction={'exact_match':P==Y,'cell_accuracy':accuracy,'changed_cell_accuracy':sum(P[i]==Y[i] for i in D)/len(D) if D else None,'false_changes':false,'false_change_rate':false/len(U) if U else None},paired_cell_accuracy_difference=accuracy-base['cell_accuracy'],paired_exact_difference=int(P==Y)-int(base['exact_match']),predictive_success=P==Y)
 return out
