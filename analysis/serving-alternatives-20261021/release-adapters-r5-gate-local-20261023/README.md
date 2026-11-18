# Narrow post-identity query-gate correction

LOCAL CPU ONLY. R4 and its submitted/reviewer proofs, scope bindings and late-gate
witness remain unchanged and are bound by PRIOR-R4-SNAPSHOT.json. No experiment
packet, freeze, assignment, GPU/network work, staging or commit is included.

The runtime delta is exactly two lines in capture.py after final ready-identity
validation and immediately before the gate write:

```python
checkpoint()
shutdown.before_deadline(work_end)
os.write(gate_write, b'G')
```

The first rechecks ownership/deadline/cancellation after validation. The last
check includes time consumed by those checks/callbacks, immediately before gate
admission. Original work/absolute ends, cleanup reserve, controller pidfd
ownership, retained-handle failure behavior, filter, scope bindings, resource
thresholds and cache policy are unchanged. Syscall/scheduling bounds remain
cooperative; no atomic kernel deadline or controller-loss guarantee is implied.

Preserved baseline-red evidence: original capture SHA from R4 plus new tests,
2 expected FAILs for successful validation crossing deadline/cancellation and
1 ordinary-success PASS. The old implementation wrote G in both prohibited
cases. The original owner's cleanup/failure receipts and explicit test rescue
remain evidence; late admission was never a successful cleanup verdict.

After correction: qualified-r2/PROOF.json binds64tests PASS14.929s and429 checked
raw references. All60 previous tests (including27 accepted core regressions)
remain. Added cases cover deadline crossing during successful identity
validation, cancellation crossing there, ordinary success, and deadline crossing
inside the final checkpoint's callback. Each negative case requires no execution
authorization and no query marker; cancellation within budget is cleaned, while
deliberately overdue validation retains the original failed cleanup accounting.
Test-only rescue never qualifies an overdue cleanup.

verify.py additionally checks the complete preserved R4 source/document/evidence
snapshot before/after the suite. All908 trial10 bindings remain exact. Adapters,
controller core, registration, runtime scope, helper, filter and existing tests
are unchanged apart from the four appended tests; LOCAL-DELTA binds the exact
patch and qualification inventory.

```bash
timeout 600s python3 verify.py --output fresh-independent-review
```

Real nvidia-smi compatibility, actual Kaggle/native integration and GPU release
remain unproved;140GiB provenance/eligibility remains independently unresolved.
The earlier release-only proposal is not constructed/frozen/assigned here.
