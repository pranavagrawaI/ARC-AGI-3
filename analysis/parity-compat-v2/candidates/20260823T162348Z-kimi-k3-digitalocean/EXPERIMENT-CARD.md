# Kimi K3 compatibility authorization candidate

**Question:** Can the 2.8T-parameter `moonshotai/kimi-k3`, with reasoning enabled, pass the frozen Native and sterile-CodeAct compatibility prerequisites?

**Route:** OpenRouter, exact `digitalocean` endpoint, fallbacks disabled. Requested client model `openai/moonshotai/kimi-k3`; expected reported model `moonshotai/kimi-k3` and provider `DigitalOcean`.

**Contrast:** Matched Native and CodeAct arms at each gate; both arms complete before advancement is judged.

**Caps:** 3 calls per arm/gate, 30 calls total, 16,384 input and 2,048 output tokens per call, 300 seconds per arm/gate, USD 3.00 total.

**Stop:** First failed pair, integrity/route/provider failure, any cap, or completed ladder.

**Advance:** Only if both arms pass every prerequisite gate. This does not authorize the 12-packet cognitive screen or live ARC.

**Status:** Frozen candidate only. No provider call is authorized until Pranav explicitly approves this exact manifest hash and cap.
