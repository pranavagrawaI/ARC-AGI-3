result={'variant':'planner','complete':True}

def successors(state):
    if state<3: yield 'advance',state+1
def key(state): return state
def goal(state): return state==3
plan=bounded_search(0,successors,state_key=key,is_goal=goal,max_depth=4)
assert plan['status']=='FOUND' and plan['depth']==3
result['search_status']=plan['status']
save_data('fixture_result',result)
print('NATIVE_FIXTURE_OK')
