# DeepSeek V4 Flash DeepInfra compatibility authorization

**Question:** Can the smaller DeepSeek V4 Flash model pass the unchanged V4 Native and sterile-CodeAct compatibility ladder?

**Contrast:** Model substitution only versus Kimi K3 V4. The prior official-endpoint attempt was provider-censored; this is a fresh exact `deepinfra/fp8` authorization, not a retry of that run.

**Route:** OpenRouter `deepinfra/fp8`, fallbacks disabled; reasoning enabled.

**Caps:** 3 calls per arm/gate, 30 calls total, 16,384 input and 2,048 output tokens per call, 300 seconds per arm/gate, USD 3.00 total.

**Stop:** Complete both arms of the reached gate; stop at first failed pair, integrity/provider failure, or cap.
