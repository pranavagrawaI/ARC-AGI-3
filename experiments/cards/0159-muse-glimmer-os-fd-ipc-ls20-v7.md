# 0159 Muse OS-FD IPC LS20 v7
Question: Can 0159 execute LS20 when protocol output bypasses mutable Python stdout state?
Contrast: v6 startup stream object versus a duplicated startup OS file descriptor.
Prediction: Every completed model turn reaches the host as one prefixed decision or error.
Measure: levels, actions, calls, budget/commit events, IPC responses, usage, stop, and hashes.
Cap: 44 ARC actions; one action per turn; 12 calls per turn: at most 11 analysis plus one `submit_action`; 8,192 output tokens per call.
Stop: level completion, game over/reset, cap, missing/illegal commitment, provider/worker/sandbox/IPC failure, or uncertain ARC action.
Invalidity: Wrong model/provider/source, analysis tool in commitment, host-selected action, mutable evidence, unenforced sandbox, resume, or missing successor.
Result to action: Preserve the run; fix technical/interface failures only. Add neither game semantics nor host fallback.
