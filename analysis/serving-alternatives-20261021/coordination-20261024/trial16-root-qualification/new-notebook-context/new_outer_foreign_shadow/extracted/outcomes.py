"""Candidate-alone interpretation only. Unknown failures are inconclusive."""
import math

SUBSTANTIVE = {'STARTUP_NOT_READY_AT_CAP', 'BOUND_SERVER_EXIT', 'OBSERVED_OOM',
               'COMPLETE_HTTP_FAILURE', 'COMPLETE_API_CONTRACT_FAILURE',
               'COMPLETE_WITHIN_ARM_GATE_FAILURE'}


def classify(evidence):
    result = dict(status='INVALID_INCONCLUSIVE', U_resp=None, adoption=False,
                  contrast='none', historical_comparison=False)
    if not all(evidence.get(key) is True for key in
               ('integrity_complete', 'identity_bound', 'cleanup_complete')):
        return result
    failure = evidence.get('failure')
    if failure is not None:
        if failure.get('kind') in SUBSTANTIVE and failure.get('raw_verified') is True:
            result['status'] = 'CLOSE_TESTED_BOUNDED_CONFIGURATION'
        return result
    rate = evidence.get('U_resp')
    if (evidence.get('within_arm_valid') is True and type(rate) in (float, int)
            and math.isfinite(rate) and rate >= 0):
        result.update(status='CANDIDATE_ALONE_WORKLOAD_FEASIBLE', U_resp=rate)
    return result
