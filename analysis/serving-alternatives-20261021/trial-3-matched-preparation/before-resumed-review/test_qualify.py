import copy
import unittest
from qualify import CONTRACT, compare, occupancy


def trace(low=8, high=9, fraction=.72, duration=900):
    events = [(0., "trace_start", None)]
    events += [(0., "decode_start", str(i)) for i in range(low)]
    current="0"
    for index, t in enumerate(range(5,duration,10)):
        events.append((float(t),"decode_end",current))
        current="base-replacement-"+str(index)
        events.append((float(t),"decode_start",current))
    for cycle in range(duration // 100):
        events.append((cycle*100., "decode_start", "extra"+str(cycle)))
        events.append((cycle*100.+100*fraction, "decode_end", "extra"+str(cycle)))
    events += [(float(t), "heartbeat", None) for t in range(duration+1)]
    events.append((float(duration),"decode_end","1"))
    events.append((float(duration), "trace_stop", None))
    events.sort(key=lambda x: x[0])
    return [{"t": t, "event": k, "pid": 1, "boot_id":"fixture-boot", "seq": i, **({"request_id": rid} if rid is not None else {})}
            for i, (t, k, rid) in enumerate(events)]


def arm(tokens=1000):
    events=trace()
    ended=[x for x in events if x["event"]=="decode_end"]
    completed={x["request_id"] for x in ended}
    attempts=[dict(request_id=x["request_id"],native_request_id=x["request_id"],sent_t=x["t"],
                   **({"terminal":"cutoff_pending","cutoff_t":900.} if x["request_id"] not in completed else {}))
              for x in events if x["event"]=="decode_start"]
    return dict(contract=CONTRACT, window=[0,900], native_events=events,
                native_adapter_qualified=True, same_host_monotonic_verified=True, full_image_fidelity=True,
                all_corpus_preprocess_fits=True, raw_responses_complete=True, identity_verified=True,
                warmup_complete=True, preemptions=0, retractions=0, restarts=0, trace_drops=0, exhausted_games=0,request_errors=0,
                attempts=attempts,
                responses=[dict(request_id=x["request_id"],native_request_id=x["request_id"],status=200,
                                done_t=x["t"],usage=dict(completion_tokens=tokens)) for x in ended])


def pair():
    return dict(control_legacy_gauge=dict(mean=8.97,p9=.72), same_corpus_manifest=True,
                same_payload_policy=True, same_image_bytes=True, same_processor_geometry=True,
                prompt_bin_tvd=0., image_bin_tvd=0., cap_bin_tvd=0., early_stop_rate_difference=0.,
                mean_completion_fraction_ratio=1.)


class QualificationTests(unittest.TestCase):
    def test_118_closes_only_tested_config(self):
        self.assertEqual(compare(arm(),arm(1180),pair())["verdict"], "CLOSE_TESTED_CONFIG")

    def test_120_permits_only_quality(self):
        self.assertEqual(compare(arm(),arm(1200),pair())["verdict"], "QUALITY_INVESTIGATION_ONLY")

    def test_integral_includes_unfinished_streams(self):
        got=occupancy(trace(),0,900)
        self.assertAlmostEqual(got["mean"],8.72)
        self.assertAlmostEqual(got["p9"],.72)
        self.assertAlmostEqual(got["stream_seconds"],7848.)

    def test_http_ten_cannot_rescue_native_eight(self):
        b=arm();b["native_events"]=trace(low=7);b["http_pending_mean"]=10
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_completed_request_stats_are_insufficient(self):
        b=arm();b["native_events"]=[]
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_missing_usage_blocks_ratio(self):
        b=arm();del b["responses"][0]["usage"]
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_sequence_loss_and_clock_gaps_fail(self):
        for change in ("seq", "time"):
            b=arm()
            if change=="seq": b["native_events"][10]["seq"]+=1
            else: b["native_events"]=b["native_events"][:10]+b["native_events"][20:]
            self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_prefill_inclusive_gauge_cannot_substitute(self):
        b=arm();b["contract"]="vllm:num_requests_running"
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_full_images_and_processor_geometry_required(self):
        for key in ("same_image_bytes","same_processor_geometry"):
            p=pair();p[key]=False
            self.assertEqual(compare(arm(),arm(),p)["verdict"],"FEASIBILITY_ONLY")

    def test_preemption_and_unqualified_adapter_fail(self):
        for key,value in (("preemptions",1),("native_adapter_qualified",False)):
            b=arm();b[key]=value
            self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_workload_mix_and_legacy_invalid_fail(self):
        for key,value in (("prompt_bin_tvd",.11),("control_legacy_gauge",dict(mean=8.05,p9=.43))):
            p=pair();p[key]=value
            self.assertEqual(compare(arm(),arm(),p)["verdict"],"FEASIBILITY_ONLY")

    def test_boundary_response_excluded_and_rate_normalized(self):
        a=arm();a["responses"][-1]["usage"]["completion_tokens"]=999999
        got=compare(a,arm(),pair())["control"]
        self.assertEqual(got["completion_tokens"],99000)
        self.assertAlmostEqual(got["tokens_per_running_stream_second"],99000/7848.)

    def test_duplicate_and_unmatched_responses_fail(self):
        for mode in ("duplicate","unmatched","native_mismatch"):
            b=arm()
            if mode=="duplicate":b["responses"]*=2
            elif mode=="unmatched":b["responses"][0]["request_id"]="unknown"
            else:b["responses"][0]["native_request_id"]="unknown"
            self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_silent_request_failure_or_missing_terminal_fails(self):
        for mode in ("error","missing"):
            b=arm()
            if mode=="error":b["request_errors"]=1
            else:b["attempts"].append(dict(request_id="lost",native_request_id="lost",sent_t=20.))
            self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_interior_boundaries_or_boot_change_fail(self):
        for mode in ("boundary","boot"):
            b=arm()
            if mode=="boundary":b["native_events"][20]["event"]="trace_start"
            else:b["native_events"][20]["boot_id"]="another-boot"
            self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_nonfinite_response_time_fails(self):
        b=arm();b["responses"][0]["done_t"]=float("nan")
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_explicit_end_cutoff_pending_is_preserved(self):
        b=arm();baseline=sum(x.get("terminal")=="cutoff_pending" for x in b["attempts"])
        b["attempts"].append(dict(request_id="unfinished",native_request_id="unfinished",sent_t=890.,
                                        terminal="cutoff_pending",cutoff_t=900.))
        got=compare(arm(),b,pair())
        self.assertEqual(got["verdict"],"CLOSE_TESTED_CONFIG")
        self.assertEqual(got["candidate"]["unresolved_at_cutoff"],baseline+1)

    def test_phantom_native_stream_fails(self):
        b=arm();b["attempts"]=[x for x in b["attempts"] if x["request_id"]!="2"]
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")

    def test_native_first_output_cannot_precede_http_send(self):
        b=arm();rid=b["responses"][0]["request_id"];b["responses"][0]["done_t"]=6.
        next(row for row in b["attempts"] if row["request_id"]==rid)["sent_t"]=5.5
        self.assertEqual(compare(arm(),b,pair())["verdict"],"FEASIBILITY_ONLY")


if __name__=="__main__":unittest.main()
