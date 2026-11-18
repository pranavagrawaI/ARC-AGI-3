import copy,unittest
import candidate_evidence as e
class ReplayJoin(unittest.TestCase):
    def fixture(self):
        identity={'pid':101,'starttime':123}
        events=[dict(event='ready',arm='candidate',t=1),dict(event='candidate_server_bound',identity=identity,session_id='CPU',gpu_uuid='GPU-CPU',t=2),dict(event='spawn',phase='candidate.replay',pid=202,pgid=202,t=3,deadline=1513),dict(event='exit',phase='candidate.replay',pid=202,t=1504),dict(event='cleanup',pgid=202,remaining=[],t=1505),dict(event='candidate_cleanup_start',cleanup_start=1506)]
        return events,{'pid':202},{'origin':3.1,'end':1503.2},identity,{'run_id':'CPU'},'GPU-CPU'
    def test_actual_child_join_and_wrong_pid_lateness_cleanup_mutations(self):
        args=self.fixture();e.replay_join(*args)
        mutations=[lambda a:a[1].update(pid=203),lambda a:a[0][3].update(t=1514),lambda a:a[0][4].update(remaining=[202]),lambda a:a[0][4].update(t=1507),lambda a:a[0][1].update(identity={'pid':101,'starttime':124})]
        for change in mutations:
            value=copy.deepcopy(args);change(value)
            with self.assertRaises(ValueError):e.replay_join(*value)
