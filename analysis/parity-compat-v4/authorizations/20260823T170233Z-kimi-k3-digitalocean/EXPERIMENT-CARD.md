# Kimi K3 transport-congruence compatibility authorization

**Question:** Does structured CodeAct action history repair Kimi K3's post-query transport failure while preserving the frozen V3 contract?

**Intervention:** V4 structured historical `execute_python` tool calls and paired tool results versus preserved V3 textual history. No prose execution and no other model-facing change.

**Route:** OpenRouter `digitalocean`, fallbacks disabled; reasoning enabled.

**Caps:** 3 calls per arm/gate, 30 calls total, 16,384 input and 2,048 output tokens per call, 300 seconds per arm/gate, USD 3.00 total.

**Stop:** Complete both arms of the reached gate; stop at first failed pair, integrity/provider failure, or cap.
