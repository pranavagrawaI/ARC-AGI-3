# probe-b-correspondence-20260819T015200Z
Question: Can muse-glimmer-30b name which movable body can exactly cover ar25's target, given the frame directly?
Posture: diagnosis
Contrast: raw opening frame vs pre-segmented bitmaps of the three structures
Prediction: raw <=1/4 correct; segmented >=3/4, since there body 4's bitmap is byte-identical to the target's
Measure: responses.jsonl `verdict` per trial; correct = names colour 4 in its final answer. 4 trials/condition, temp 1.0
Cap: 8 calls, 8000 max_tokens each, 40 min, no ARC actions, no game session
Stop: all 8 complete, or 2 consecutive API errors
Invalid if: prompt names the answer, segmentation drops/merges a body, >1 ambiguous per condition, or any trial truncated with null content
Result map: segmented high & raw low -> attention failure, fix by forcing a segmented correspondence pass (Layers 1-2). both low -> capability gap, program must enumerate candidates (Layer 3). both high -> neither; the loop never asked
Outputs: scratchpad/probeB/responses.jsonl
