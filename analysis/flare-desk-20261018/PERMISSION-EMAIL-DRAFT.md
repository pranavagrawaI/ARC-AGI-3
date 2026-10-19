# Draft: written permission request to Local Inference Lab (LIL-1.0 §2.4)

Not sent. Pranav sends it, or doesn't. The address is whatever contact the lab lists on Hugging Face or GitHub
(local-inference-lab); I did not look one up or send anything.

---

Subject: Permission request: private Kaggle copy of Qwen3.8-Flash-Next-FLARE (gb202) for an offline competition

Hello,

I'm competing in the ARC Prize 2026 (ARC-AGI-3) Kaggle competition. Submissions run on Kaggle's RTX PRO 6000
Blackwell (96 GB) with internet disabled, so any model has to be attached to the notebook as a Kaggle dataset
or notebook output beforehand.

I'd like to evaluate Flare with your gb202 package (`qwen3.8-flash-next-tp1-c16-mtp5.safetensors`) as the
serving runtime for my agent. To do that, I would keep a byte-identical copy of the package and the `flare`
binary in private Kaggle storage on my own account. It would never be public or shared with any other user,
and it would be used only by my notebooks and by my competition submissions, which Kaggle runs on my
behalf without exposing the attached files.

My reading of LIL-1.0 §2.2(b) is that this is "storage that only You control". Since Kaggle is a third-party
host, though, I'd rather have your written confirmation under §2.4 than rely on that reading. Could you confirm
that this use is permitted? If you'd like it attributed or limited in any way, I'm happy to follow your terms.

If it's useful to you, I'm also glad to share throughput and tool-call results from Kaggle's RTX PRO 6000
(single GPU, 10-16 concurrent long-context agent streams).

Thank you,
Pranav Agrawal
