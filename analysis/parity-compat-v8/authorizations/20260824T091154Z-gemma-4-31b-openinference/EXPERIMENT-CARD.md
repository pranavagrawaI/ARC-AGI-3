# Gemma 4 31B V8 compatibility authorization

**Question:** Can the smaller Gemma 4 31B model use the frozen Native and sterile CodeAct protocols through the compatibility ladder?

**Contrast:** Frozen V8 with Gemma 4 31B on the exact OpenInference BF16 route versus the preserved DeepSeek V8 run. Interface, tasks, evidence, semantics, authority, and ceilings are unchanged; model and provider route change.

**Prediction:** Both arms clear termination. Passing metadata establishes protocol compatibility; representation outcomes are interpreted separately. No claim depends on matching DeepSeek sampling behavior.

**Measure:** Gate outcomes; typed returns; capability attempts and disclosures; transport adherence; route; tokens; cost; integrity.

**Caps:** 3 calls per arm/gate, 30 calls total, 16,384 input and 2,048 output tokens per call, 300 seconds per arm/gate, USD 3.00 total.

**Stop:** Finish both arms of the reached gate; stop at first failed pair, provider/integrity failure, or cap. Raw-grid input-cap failure censors that pair rather than diagnosing model reasoning.
