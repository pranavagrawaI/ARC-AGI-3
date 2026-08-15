# 0159 Muse stable-IPC commitment LS20 v6
Question: Can 0159 execute LS20 without tool exhaustion or redirected-stdout deadlock?
Contrast: v5 with worker protocol output bound to startup stdout rather than ambient stdout.
Prediction: Every completed model turn reaches the host as one prefixed decision or error.
Measure: levels, actions, calls, budget/commit events, IPC responses, usage, stop, and hashes.
Cap: 44 ARC actions; one action per turn; 12 calls per turn: at most 11 analysis plus one `submit_action`; 8,192 output tokens per call.
Stop: level completion, game over/reset, cap, missing/illegal commitment, provider/worker/sandbox/IPC failure, or uncertain ARC action.
Invalidity: Wrong model/provider/source, analysis tool in commitment, host-selected action, mutable evidence, unenforced sandbox, resume, or missing successor.
Result to action: Preserve the run; fix technical/interface failures only. Add neither game semantics nor host fallback.
