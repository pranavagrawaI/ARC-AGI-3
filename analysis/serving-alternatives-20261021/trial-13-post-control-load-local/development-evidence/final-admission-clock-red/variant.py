def candidate_gate(self,end=None,fresh=None):
    # Identity checks may block: sample time only AFTER the final checks.
    now=time.monotonic()
    living=shutdown.owned(self.binding)
    self.check_observer()
    phases=[g for g in self.groups if self.observer is None or g!=self.observer.pid]
    require(now<min(self.work_end,end if end is not None else self.work_end),'candidate work deadline')
    admission.require_candidate(fresh if fresh is not None else self.candidate_admission,now,
        self.config['admission']['maximum_age_seconds'],living,list(capture.RETAINED),phases)
