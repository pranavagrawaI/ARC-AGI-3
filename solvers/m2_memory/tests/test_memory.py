"""Saved-data and observation contracts through actual sandbox processes."""
import json
from pathlib import Path
from types import SimpleNamespace
from test_basics import SandboxCase, STATE, extracted


class SavedData(SandboxCase):
    def test_checkpoint_survives_timeout_and_is_detached(self):
        store = {}
        result = self.run_code('x={"a":[1,2]}; save_data("map",x); x["a"].append(9)\nwhile True: pass',
                               saved_data=store, timeout=1)
        self.assertTrue(result['timed_out'])
        result = self.run_code('r=load_data("map"); r["value"]["a"].clear(); print(load_data("map")); print(list_data())', saved_data=store)
        self.assertFalse(result['error'], result)
        self.assertEqual(store['map']['value'], {'a':[1,2]})
        self.assertEqual(store['map']['saved_step'], 0)
        self.assertEqual(store['map']['saved_level'], 1)
        self.assertIn("'map'", result['stdout'])

    def test_failed_saves_preserve_previous_and_limits_allow_replacement(self):
        store={}
        self.assertFalse(self.run_code('save_data("x", [1])',saved_data=store)['error'])
        for value in ['(1,2)', '{1:2}', '{"x":float("nan")}', '"x"*262145', 'set([1])']:
            with self.subTest(value=value):
                result=self.run_code('save_data("x",'+value+')',saved_data=store)
                self.assertIn('ValueError',result['error'])
                self.assertEqual(store['x']['value'],[1])
        result=self.run_code('for i in range(63): save_data(str(i), i)\nsave_data("extra",0)',saved_data=store)
        self.assertIn('64 items',result['error'])
        result=self.run_code('save_data("x",[2]); delete_data("0"); save_data("extra",3)',saved_data=store)
        self.assertFalse(result['error'],result)
        self.assertEqual(store['x']['value'],[2])
        self.assertEqual(len(store),64)

    def test_total_byte_limit_and_deletion(self):
        store={}
        result=self.run_code('for i in range(5): save_data(str(i), "x"*250000)',saved_data=store)
        self.assertIn('1 MiB', result['error'])
        self.assertEqual(set(store),{'0','1','2','3'})
        result=self.run_code('delete_data("0"); save_data("4", "x"*250000); print(delete_data("missing"))', saved_data=store)
        self.assertFalse(result['error'],result)
        self.assertEqual(result['stdout'].strip(),'False')

    def test_action_updates_provenance_without_rewriting_old_snapshot(self):
        store={}
        after={'current_frame':{**STATE['current_frame'],'step':8,'level':2},'history':[]}
        def handler(actions, **kwargs):
            return {'action_result':{'executed':True},'state':after}
        result=self.run_code('save_data("before", [1]); action("UP"); save_data("after", [2])',
                             saved_data=store,handler=handler)
        self.assertFalse(result['error'],result)
        self.assertEqual((store['before']['saved_step'],store['before']['saved_level']),(0,1))
        self.assertEqual((store['after']['saved_step'],store['after']['saved_level']),(8,2))
        self.assertTrue(result['action_results'][0]['executed'])

    def test_actual_agent_session_boundary_clears_data(self):
        host=SimpleNamespace(_session_runtime_dir=Path('/game/a'),_python_saved_data={'x':{'value':1}})
        fn=extracted('_ensure_session',{'Path':Path,'_persistent_functions':lambda:False,'_empty_world_model':dict})
        fn(host,Path('/game/a/state.json'))
        self.assertIn('x',host._python_saved_data)
        fn(host,Path('/game/b/state.json'))
        self.assertEqual(host._python_saved_data,{})

    def test_retained_helper_can_read_saved_data(self):
        store={}
        first=self.run_code('save_data("rule", {"offset":3})\ndef offset(x): return x+load_data("rule")["value"]["offset"]',
                            saved_data=store,kept_functions=[])
        self.assertFalse(first['error'],first)
        self.assertEqual([x['name'] for x in first['keepable_functions']],['offset'])
        second=self.run_code('print(offset(4))',saved_data=store,
                             kept_functions=[x['source'] for x in first['keepable_functions']])
        self.assertFalse(second['error'],second)
        self.assertEqual(second['stdout'].strip(),'7')

    def test_agent_dispatch_carries_saved_data_across_calls(self):
        ns={'Path':Path,'Any':object,'json':json,
            '_ToolDispatchResult':lambda content,**kw:SimpleNamespace(content=content,**kw),
            'load_runtime_state':lambda path:(None,[]),'_normalize_valid_actions':list,
            '_ascii_frame_view_payload':lambda f:STATE['current_frame'],'_ascii_history_view_payload':lambda h:[],
            '_get_env_bool':lambda *args:False,'_animation_enabled':lambda:False}
        from test_basics import sandbox
        ns['run_sandboxed_python']=sandbox.run_sandboxed_python
        host=SimpleNamespace(_ensure_session=lambda p:None,_current_valid_actions=['UP'],_last_action_call_result={},
            _step_env_callback=None,_noop_repeat_guard=None,_repeat_state_seen=set(),_python_timeout=3,
            _retained_sources=lambda:None,_record_retained_functions=lambda *a:None,
            _render_tool_payload=lambda p,**kw:json.dumps(p),_last_step_summary=None)
        fn=extracted('_run_python_tool',ns)
        first=fn(host,Path('/unused'),{'code':'save_data("plan", ["UP"])'})
        self.assertNotIn('error',json.loads(first.content))
        second=fn(host,Path('/unused'),{'code':'print(load_data("plan")["value"])'})
        self.assertEqual(json.loads(second.content)['stdout'].strip(),"['UP']")


class Views(SandboxCase):
    def test_views_use_numeric_colors_row_column_and_are_detached(self):
        state={'current_frame':{'grid':[[2,1,2],[0,2,1]],'ascii':'bwb\nWbw','shape':[2,3],'step':4,'level':1},'history':[]}
        code='''print(current_frame.rows)
print(current_frame.grid)
c = current_frame.counts; c.clear()
print(current_frame.counts)
print(current_frame.cells(2)); print(current_frame.cells(15))
'''
        result=self.run_code(code,state=state)
        self.assertFalse(result['error'],result)
        self.assertEqual(result['stdout'].splitlines(),["('bwb', 'Wbw')",'((2, 1, 2), (0, 2, 1))',
            '{2: 3, 1: 2, 0: 1}','((0, 0), (0, 2), (1, 1))','()'])

    def test_observation_refresh_preserves_old_views(self):
        after={'current_frame':{**STATE['current_frame'],'grid':[[1,1]],'ascii':'ww','step':1},'history':[]}
        def handler(actions,**kwargs):return {'action_result':{'executed':True},'state':after}
        result=self.run_code('old=current_frame; print(old.cells(0)); action("UP"); print(current_frame.cells(0)); print(old.cells(0))',handler=handler)
        self.assertFalse(result['error'],result)
        self.assertEqual(result['stdout'].strip().splitlines(),['((0, 0),)','()','((0, 0),)'])

    def test_history_alias_and_animation_mapping(self):
        state={**STATE,'history':[{'action':'UP','frame':STATE['current_frame'],'result':{}}]}
        result=self.run_code('print(history[0].step,history[0].level); print(list(last_animation_timeline.items())); print(list(last_animation_timeline.values()))',
            state=state,animation_handler=lambda request:{'steps':[1,2]})
        self.assertFalse(result['error'],result)
        self.assertEqual(result['stdout'].splitlines(),['0 1',"[('steps', [1, 2])]",'[[1, 2]]'])
