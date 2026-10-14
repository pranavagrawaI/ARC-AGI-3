"""Public search contracts, checked against independent small-graph BFS."""
from collections import deque
import importlib.util
from pathlib import Path
import random
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'src/inference/utils/bounded_search.py'
spec = importlib.util.spec_from_file_location('search_under_test', SOURCE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
search = module.bounded_search


def solve(graph, target, **kwargs):
    return search('S', lambda s: [(v, v) for v in graph.get(s, [])],
                  state_key=lambda s:s, is_goal=lambda s:s==target, **kwargs)


class SearchTests(unittest.TestCase):
    def test_initial_goal_and_unit_shortest_route(self):
        r = solve({'S':['A'], 'A':['G']}, 'S')
        self.assertEqual((r['status'], r['path'], r['expanded']), ('FOUND', [], 0))
        r = solve({'S':['A','B'], 'A':['C'], 'C':['G'], 'B':['G']}, 'G')
        self.assertEqual((r['status'], r['path'], r['depth']), ('FOUND',['B','G'],2))

    def test_horizon_and_graph_exhaustion_are_distinct(self):
        graph = {'S':['A'], 'A':['G']}
        self.assertEqual(solve(graph, 'G', max_depth=1)['status'], 'HORIZON_EXHAUSTED')
        self.assertEqual(solve(graph, 'G', max_depth=2)['status'], 'FOUND')
        self.assertEqual(solve({'S':['A'], 'A':['S']}, 'G')['status'], 'GRAPH_EXHAUSTED')
        self.assertEqual(solve(graph, 'G', max_depth=0)['depth_frontier_states'], 1)

    def test_reopening_a_shallower_state_keeps_real_predecessors(self):
        graph={'S':['A','B'],'A':['C'],'C':['X'],'B':['X'],'X':['G']}
        h=lambda s:100 if s=='B' else 0
        r=solve(graph,'G',heuristic=h,max_depth=3)
        self.assertEqual(r['path'], ['B','X','G'])
        self.assertEqual(r['depth'],len(r['path']))

    def test_reopened_cut_is_not_an_unexplored_frontier(self):
        graph={'S':['A','B'],'A':['C'],'C':['X'],'B':['X']}
        r=solve(graph,'G',heuristic=lambda s:100 if s=='B' else 0,max_depth=3)
        self.assertEqual((r['status'],r['depth_frontier_states']),('GRAPH_EXHAUSTED',0))

    def test_heap_never_compares_model_states(self):
        r=search({'id':0},lambda s:[('left',{'id':1}),('right',{'id':2})] if s['id']==0 else [],
                 state_key=lambda s:s['id'],is_goal=lambda s:s['id']==2)
        self.assertEqual(r['path'], ['right'])

    def test_rejections_and_duplicate_cycles(self):
        r=search(0,lambda s:[('blocked',None),('loop',s)],state_key=lambda s:s,is_goal=lambda s:False)
        self.assertEqual((r['status'],r['rejected'],r['generated']),('GRAPH_EXHAUSTED',1,2))

    def test_node_and_expansion_caps(self):
        for caps,reason in [({'max_nodes':1},'NODE_LIMIT'),({'max_expanded':1},'EXPANSION_LIMIT')]:
            r=solve({'S':['A'],'A':['G']},'G',**caps)
            self.assertEqual((r['status'],r['reason']),('RESOURCE_LIMIT',reason))
        r=solve({'S':['A'],'A':['G']},'G',max_expanded=2)
        self.assertEqual(r['status'],'FOUND')

    def test_zero_time_calls_no_callback(self):
        def forbidden(s):
            raise AssertionError('called despite zero budget')
        r=search(0,forbidden,state_key=forbidden,is_goal=forbidden,max_seconds=0)
        self.assertEqual((r['status'],r['reason']),('RESOURCE_LIMIT','TIME_LIMIT'))

    def test_callback_return_after_deadline_cannot_publish_found(self):
        def slow_goal(s):
            time.sleep(.03)
            return True
        r=search(0,lambda s:[],state_key=lambda s:s,is_goal=slow_goal,max_seconds=.01)
        self.assertEqual((r['status'],r['reason']),('RESOURCE_LIMIT','TIME_LIMIT'))

    def test_infinite_duplicate_generator_is_timed(self):
        def infinite(s):
            while True:
                yield 'loop',s
        r=search(0,infinite,state_key=lambda s:s,is_goal=lambda s:False,max_seconds=.01)
        self.assertEqual((r['status'],r['reason']),('RESOURCE_LIMIT','TIME_LIMIT'))

    def test_callback_errors_are_reported(self):
        r=search(0,lambda s:[('x',1)],state_key=lambda s:[],is_goal=lambda s:False)
        self.assertEqual((r['status'],r['callback']),('CALLBACK_ERROR','state_key'))
        r=search(0,lambda s:[('malformed',)],state_key=lambda s:s,is_goal=lambda s:False)
        self.assertEqual((r['status'],r['callback']),('CALLBACK_ERROR','successors'))
        r=search(0,lambda s:[],state_key=lambda s:s,is_goal=lambda s:1)
        self.assertEqual((r['status'],r['callback']),('CALLBACK_ERROR','is_goal'))

    def test_invalid_heuristics_and_caps(self):
        for h in [float('nan'),float('inf'),-1,True,'wrong']:
            r=solve({},'G',heuristic=lambda s:h)
            self.assertEqual((r['status'],r['callback']),('CALLBACK_ERROR','heuristic'))
        for caps in [{'max_nodes':0},{'max_expanded':True},{'max_depth':-1},
                     {'max_seconds':float('nan')},{'max_seconds':-1},{'max_seconds':True}]:
            with self.assertRaises(ValueError):
                solve({},'G',**caps)
        with self.assertRaises(TypeError):
            search(0,None,state_key=lambda s:s,is_goal=lambda s:False)

    def test_random_graphs_match_independent_bfs(self):
        rng=random.Random(917)
        for _ in range(200):
            n=rng.randint(2,16)
            graph={i:[j for j in range(n) if rng.random()<.17] for i in range(n)}
            q=deque([(0,0)]);seen={0};distance=None
            while q:
                s,d=q.popleft()
                if s==n-1:
                    distance=d;break
                for nxt in graph[s]:
                    if nxt not in seen:
                        seen.add(nxt);q.append((nxt,d+1))
            r=search(0,lambda s:[(v,v) for v in graph[s]],state_key=lambda s:s,is_goal=lambda s:s==n-1)
            if distance is None:
                self.assertEqual(r['status'],'GRAPH_EXHAUSTED')
            else:
                self.assertEqual((r['status'],r['depth']),('FOUND',distance))
                state=0
                for action in r['path']:
                    self.assertIn(action,graph[state]);state=action
                self.assertEqual(state,n-1)


if __name__=='__main__':
    unittest.main()
