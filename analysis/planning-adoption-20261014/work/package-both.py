result={'variant':'both','complete':True}

def successors(state):
    if state<3: yield 'advance',state+1
def key(state): return state
def goal(state): return state==3
plan=bounded_search(0,successors,state_key=key,is_goal=goal,max_depth=4)
assert plan['status']=='FOUND' and plan['depth']==3
result['search_status']=plan['status']

def observe(frame): return {'grid':[list(row) for row in frame.grid]}
def predict(before,action): return before
def grid_key(state): return tuple(tuple(row) for row in state['grid'])
probe=probe_transition({'action':'MOUSE','row':57,'col':36},observe=observe,predict=predict,state_key=grid_key,name='fixture_probe')
assert probe['status'] in ('MATCH','MISMATCH','TERMINAL','REFUSED')
result['probe_status']=probe['status']
save_data('fixture_result',result)
print('NATIVE_FIXTURE_OK')
