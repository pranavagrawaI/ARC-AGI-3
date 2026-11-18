# Preserved review failure and test-only correction

The original root64-test run was not a full pass. Its first returned release result
was lost when teardown hit UnsafeIdentity; it cannot be reconstructed from the
isolated diagnostic pass. PRIOR-R5-SNAPSHOT binds both original log and separate
diagnostic. No runtime identity, deadline, resource, filter or scope gate changes.

The source fixture normally responds to frontend TERM and can exit while owned()
is scanning /proc, before mocked query expiry is reached. A fail-closed identity
error at that point can leave controller_query_failure unset. This is a supported
race mechanism, not proof of the unavailable original first result. The targeted
fallback test now uses the existing stubborn CPU frontend: TERM cannot destroy
its identity before the query-phase intervention; the original runtime fallback
still performs KILL. That separates query-phase expiry behavior from graceful
frontend exit timing without relaxing runtime gates.

Test teardown now retains verified pidfds while bound fixtures are live, kills
only those handles, and waits for pidfd exit readiness. It does not depend on an
exit-time /proc/exe read and cannot signal a reused numeric PID. Cleanup errors
are retained and raised, while RESULT-before-cleanup/RESULT/raw/CLEANUP persist
even on failure. A negative ownership fixture demonstrated missing persistence
before the correction; it now retains the primary result and explicit cleanup
error. Unknown ownership never authorizes a signal.

Canonical full65-test suite passes14.294s, including all original runtime gate
tests and27 accepted core regressions. This qualifies the revised local test
scenario; it does not overwrite the failed root run or establish GPU release.
