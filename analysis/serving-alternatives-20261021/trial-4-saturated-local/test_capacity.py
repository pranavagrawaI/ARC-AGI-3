import tempfile
import copy
import json
import unittest
from pathlib import Path
import corpus
import capacity
import fixtures


class CorpusTests(unittest.TestCase):
    def test_modified_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);(path/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest'):
                corpus.verify_corpus(path)


class WindowTests(unittest.TestCase):
    def test_window_is_derived_from_raw_phase_clock(self):
        events=[dict(kind='replay_start',t=40.),dict(kind='measure_start',t=640.),dict(kind='measure_end',t=1540.)]
        self.assertEqual(capacity.derive_window(events),(640.,1540.))


class RawReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=corpus.load();cls.reader=fixtures.Reader(cls.catalog)
        cls.base=fixtures.build(cls.catalog,cls.reader,label='baseline')

    def reduce(self,bundle=None,reader=None):
        return capacity.reduce_arm(bundle or self.base,self.catalog,reader or self.reader)

    def test_real_corpus_events_qualify_and_normal_games_finish(self):
        got=self.reduce()
        self.assertTrue(got['valid'],got.get('issues'))
        self.assertGreater(got['load']['finished_games'],0)
        self.assertEqual(got['load']['starved_s'],0)
        self.assertEqual(got['window'],[600.,1500.])
        self.assertEqual(got['completion_tokens'],618413)
        self.assertAlmostEqual(got['U_resp'],618413/900)

    def test_phase_selection_cannot_be_overridden_by_claimed_window(self):
        b=copy.deepcopy(self.base);b['window']=[700.,1600.];b['warmup_complete']=True
        for row in b['client_events']:
            if row['kind']=='measure_start':row['t']+=.1
        self.assertFalse(self.reduce(b)['valid'])

    def test_initial_holder_warmup_is_computed(self):
        b=fixtures.build(self.catalog,self.reader,label='missing-warmup',prefill=.000001,cache=0.);b['warmup_complete']=True
        got=self.reduce(b);self.assertFalse(got['valid']);self.assertIn('warmup',got['issues'][0])

    def test_wire_image_removal_is_detected_from_bytes(self):
        row=next(x for x in self.base['client_events'] if x['kind']=='send')
        ref=row['wire_ref'];value=json.loads(self.reader(ref))
        for message in value['body']['messages']:
            if isinstance(message.get('content'),list):message['content']=[x for x in message['content'] if x.get('type')!='image_url']
        actual=corpus.canonical(value)
        got=self.reduce(reader=lambda r:actual if r==ref else self.reader(r))
        self.assertFalse(got['valid']);self.assertIn('fidelity',got['issues'][0])

    def test_response_usage_null_and_missing_fields_return_invalid(self):
        response=next(x for x in self.base['client_events'] if x['kind']=='response');ref=response['response_ref']
        raw=json.loads(self.reader(ref));raw['usage']=None
        self.assertFalse(self.reduce(reader=lambda r:corpus.canonical(raw) if r==ref else self.reader(r))['valid'])
        for field in ('client_events','native_events'):
            b=copy.deepcopy(self.base);b.pop(field)
            self.assertFalse(self.reduce(b)['valid'])

    def test_duplicate_response_does_not_inflate_rate(self):
        b=copy.deepcopy(self.base);rows=b['client_events'];index=next(i for i,x in enumerate(rows) if x['kind']=='response')
        rows.insert(index+1,dict(rows[index]))
        for seq,row in enumerate(rows):row['seq']=seq
        got=self.reduce(b);self.assertFalse(got['valid']);self.assertIn('duplicate',got['issues'][0])

    def test_distinct_requests_cannot_reuse_raw_response_identity(self):
        b=copy.deepcopy(self.base)
        for row in b['native_events']:
            if row['kind']=='accept':row['response_id']='reused-response'
        def reader(ref):
            raw=self.reader(ref)
            if ref.startswith('response/'):
                value=json.loads(raw);value['id']='reused-response';return corpus.canonical(value)
            return raw
        got=self.reduce(b,reader)
        self.assertFalse(got['valid']);self.assertIn('response identity',got['issues'][0])

    def test_native_abort_or_trace_loss_is_invalid(self):
        for kind in ('abort','preempt','retract'):
            b=copy.deepcopy(self.base)
            next(x for x in b['native_events'] if x['kind']=='heartbeat')['kind']=kind
            self.assertFalse(self.reduce(b)['valid'])
        b=copy.deepcopy(self.base);b['native_events'][20]['seq']+=1
        self.assertFalse(self.reduce(b)['valid'])

    def test_native_prompt_or_geometry_mismatch_is_invalid(self):
        for key,value in (('prompt_tokens',1),('grids',[])):
            b=copy.deepcopy(self.base);row=next(x for x in b['native_events'] if x['kind']=='prefill');row[key]=value
            self.assertFalse(self.reduce(b)['valid'])

    def test_censored_native_streams_are_included(self):
        got=self.reduce();self.assertEqual(got['unfinished'],10)
        self.assertAlmostEqual(got['native_stream_seconds'],8804.6)
        self.assertAlmostEqual(got['per_stream_rate'],618413/8804.6)

    def test_sends_after_measurement_are_invalid(self):
        b=copy.deepcopy(self.base)
        # A send at the cutoff can be chronologically valid but is forbidden
        # because this local design predeclares no post-measurement requests.
        send=next(x for x in b['client_events'] if x['kind']=='send')
        b['client_events'].insert(-1,dict(send,t=1500.,rid='late'))
        for seq,row in enumerate(b['client_events']):row['seq']=seq
        self.assertFalse(self.reduce(b)['valid'])

    def test_supplied_mix_and_occupancy_claims_are_ignored(self):
        a=self.reduce();b=self.reduce()
        # Independent source rows determine the marginal distributions.
        got=capacity.mix(a['measured_rows'],b['measured_rows'])
        self.assertEqual(got,dict(prompt_tvd=0.,images_tvd=0.,cap_tvd=0.,early_stop_difference=0.,completion_fraction_ratio=1.))

    def test_histogram_bins_are_right_closed(self):
        rows=[dict(record=dict(cap=1024)),dict(record=dict(cap=1025))]
        self.assertEqual(capacity.histogram(rows,'cap',[0,1024,2048]),[.5,.5])

    def test_trim_handover_cannot_be_skipped(self):
        b=fixtures.build(self.catalog,self.reader,label='bad-handover',skip_handover=True)
        got=self.reduce(b)
        self.assertFalse(got['valid'],got.get('issues'))

    def test_real_postwindow_sends_are_rejected(self):
        b=fixtures.build(self.catalog,self.reader,label='late-sends',extra=20.)
        self.assertTrue(any(x['kind']=='send' and x['t']>=1500. for x in b['client_events']))
        got=self.reduce(b)
        self.assertFalse(got['valid'],got.get('issues'))

    def test_unverified_catalog_changes_are_rejected(self):
        bad=copy.deepcopy(self.catalog);next(iter(bad['records'].values()))['prompt']=1
        self.assertFalse(capacity.reduce_arm(self.base,bad,self.reader)['valid'])

    def test_overlapping_pair_cannot_claim_same_session_sequence(self):
        got=capacity.compare(self.base,self.base,self.catalog,self.reader)
        self.assertEqual(got['verdict'],'FEASIBILITY_ONLY')

    def test_low_native_occupancy_is_not_a_load_gate(self):
        b=fixtures.build(self.catalog,self.reader,label='half-cache-test',cache=.5)
        got=self.reduce(b)
        self.assertTrue(got['valid'],got.get('issues'))
        self.assertTrue(got['load_valid'])
        self.assertLess(got['native_mean_R'],8.0)

    def test_lag_is_a_generator_failure_not_a_native_band(self):
        b=fixtures.build(self.catalog,self.reader,label='lag-test',lag=.3)
        got=self.reduce(b)
        self.assertFalse(got['load_valid']);self.assertIn('client generator not saturated',got['issues'])

    def test_actual_corpus_depletion_is_starvation(self):
        b=fixtures.build(self.catalog,self.reader,label='actual-depletion',decode=1000.,prefill=1000.)
        got=self.reduce(b)
        self.assertFalse(got['valid']);self.assertGreater(got['load']['starved_s'],0.)
        self.assertEqual(got['load']['finished_games'],25)

    def test_nonmapping_pair_inputs_cannot_crash_reduction(self):
        got=capacity.compare(None,self.base,self.catalog,self.reader)
        self.assertEqual(got['verdict'],'FEASIBILITY_ONLY')

    def test_readmitted_holder_cannot_skip_fifo_turn_without_completion(self):
        class SkipReadmission(fixtures.Simulation):
            def admit(sim):
                if not hasattr(sim,'admitted'):sim.admitted=set();sim.skipped=False
                while sim.queue and len(sim.holders)<10:
                    game=sim.queue.pop(0);sim.holders.add(game)
                    sim.emit('client','admit',game=game)
                    if game in sim.admitted and not sim.skipped:
                        sim.skipped=True;sim.holders.remove(game);sim.queue.append(game)
                        sim.emit('client','release',game=game,reason='handover')
                    else:
                        sim.admitted.add(game)
                        fixtures.heapq.heappush(sim.callbacks,(sim.t+sim.lag,game))
        simulation=SkipReadmission(self.catalog,self.reader,label='fifo-bypass')
        bundle=simulation.run();self.assertTrue(simulation.skipped)
        got=self.reduce(bundle)
        self.assertFalse(got['valid']);self.assertIn('handover',got['issues'][0])

    def test_native_tail_overlap_invalidates_sequential_pair(self):
        control=copy.deepcopy(self.base)
        rows=control['native_events'];end=rows.pop()
        for t in range(1501,4001):rows.append(dict(end,kind='heartbeat',t=float(t)))
        rows.append(dict(end,t=4000.))
        for seq,row in enumerate(rows):row['seq']=seq
        candidate=fixtures.build(self.catalog,self.reader,label='after-client-only',origin=2000.)
        got=capacity.compare(control,candidate,self.catalog,self.reader)
        self.assertEqual(got['verdict'],'FEASIBILITY_ONLY')
        self.assertIn('pair is not fresh sequential loads',got['issues'])

    def test_fresh_sequential_fast_fixture_can_reach_quality_investigation_only(self):
        candidate=fixtures.build(self.catalog,self.reader,label='fast-pair',origin=2000.,decode=1.5)
        got=capacity.compare(self.base,candidate,self.catalog,self.reader)
        self.assertEqual(got['verdict'],'QUALITY_INVESTIGATION_ONLY',got['issues'])
        self.assertGreaterEqual(got['ratio'],1.20)


if __name__=='__main__':unittest.main()
