The frozen replay has two competing cutoff mechanisms: a read timeout derived
from the remaining window, and sequential parent-owned socket shutdown. A local
real-socket witness reproduces timeout delivery before the socket's shutdown
marker by delaying the first parent shutdown by 80ms. The frozen source fails
this witness. This proves a possible mechanism; trial9 did not record its timer
arm and its exact timer provenance remains unobserved. Trial9 stays invalid.

A second witness detects an actual socket timeout before cutoff, then delays
delivery past the parent shutdown marker. The frozen source incorrectly treats
it as intentional cancellation. Merely accepting more after-cutoff timeouts
would therefore be unsafe.

The separately identified local variant changes only replay.py and reduce_api.py:

- Parent shutdown is the sole fixed-window cancellation mechanism. Reads retain
  the existing 900s inactivity bound without shortening it to the window's end.
- Native TimeoutError is always fatal, even after an owned marker. Owned EOF,
  reset, abort and broken pipe retain the existing cancellation checks. Definite
  protocol and persistence errors retain their fatal behavior.
- HTTP receipts record actual Python socket timeout configuration, arm clocks,
  fixed cutoff, and phase at exception handling. These do not claim an exact
  kernel timer firing time or an absolute expiry at arm+900s.
- The reducer independently rejects relabeled native timeouts and requires the
  new read configuration/phase evidence. It does not reinterpret old receipts.

The local suite passes 26 tests: the real-socket race, actual inactivity expiry,
predeadline expiry with delayed handling, unchanged malformed-protocol and disk
loss regressions, EOF/trickle cutoff, admission stopping, wire capture, and raw
reducer gates. The 900s default is observed through gettimeout(); expiry tests
explicitly compress the socket bound to .05s. They do not claim a 900s elapsed
test. Existing test_replay.py and test_persistence.py are byte-identical. The
protocol fixture changes only its archived witness path; the analytical reducer
fixture adds the new receipt fields without changing prior expectations.

Reproduce from this directory:

```sh
PYTHONPATH=baseline:tests python3 -m unittest test_deadline_race.DeadlineRaceTests.test_deadline_timer_races_sequential_parent_shutdown -v
PYTHONPATH=baseline:tests python3 -m unittest test_deadline_race.DeadlineRaceTests.test_predeadline_timeout_delayed_past_owned_shutdown_is_fatal -v
PYTHONPATH=variant:tests python3 -m unittest test_deadline_race test_replay test_persistence test_protocol_cutoff test_reducer -v
python3 verify_local.py
```

The first two commands intentionally fail on the preserved baseline. They use
loopback HTTP and explicit scheduling/time compression only. Set LOCAL_EVIDENCE
to a fresh directory to preserve new raw witnesses; existing evidence is never
overwritten. LOCAL-PROOF.json binds the source, tests, receipts and unchanged
642 trial9 inputs. This is a local correction proposal, not a successor GPU
packet, assignment or launch clearance.

Collection handoff remains separate: all ten pending receipts and primary
inventory are preserved; the single partial archive has no central directory.
No archive retry was made. The frozen reducer returns FEASIBILITY_ONLY with no
ratio, the watcher is disabled, and the coordinator closed/released trial9.
