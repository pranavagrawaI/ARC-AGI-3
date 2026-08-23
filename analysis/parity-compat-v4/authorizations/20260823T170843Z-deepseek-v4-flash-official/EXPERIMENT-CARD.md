# DeepSeek V4 Flash transport-congruence compatibility authorization

**Question:** Can the smaller DeepSeek V4 Flash model pass the same frozen V4 Native and sterile-CodeAct compatibility ladder used for Kimi K3?

**Contrast:** Model substitution only: DeepSeek V4 Flash through the exact official DeepSeek endpoint versus the preserved Kimi K3 V4 run. No protocol, prompt, gate, evidence, schema, transport, or authority changes.

**Route:** OpenRouter `deepseek`, fallbacks disabled; reasoning enabled.

**Caps:** 3 calls per arm/gate, 30 calls total, 16,384 input and 2,048 output tokens per call, 300 seconds per arm/gate, USD 3.00 total.

**Stop:** Complete both arms of the reached gate; stop at first failed pair, integrity/provider failure, or cap.
