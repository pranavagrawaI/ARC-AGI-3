"""Runtime patches for serving nvidia/Qwen3.8-Flash-Next-NVFP4 (and REAP-pruned
derivatives) with vLLM — no environment modifications.

Patch 1 — PLE FP8 embedding under ModelOpt NVFP4 (vLLM 0.29 only).
Bug: vllm/models/qwen4_exp/nvidia/ple_layer.py :: _get_ple_embedding_quant_method
only selects the FP8 PLE embedding method (which registers the
`ngram_embedding.weight_scale` parameter) when the model-level quant config is
`Fp8Config`. ModelOpt NVFP4 checkpoints use `ModelOptNvFp4Config`/
`ModelOptMixedPrecisionConfig`, so the PLE n-gram table — stored FP8 with a
per-tensor scale even in NVFP4 builds — is created WITHOUT the scale parameter,
and weight loading fails with:
  ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'
Fix: wrap the selector; if it returns None and the quant config is a ModelOpt
config, use Qwen4ExpPLEFp8EmbeddingMethod for the n-gram embedding.
vLLM 0.30+ handles this natively (ngram_embedding.py dispatches
ModelOptMixedPrecisionConfig); the patch self-disables when the symbol is gone.

Patch 2 — MTP draft expert count for pruned checkpoints (vLLM 0.29/0.30+).
REAP surgery reduces the MAIN model's routed experts (text_config.num_experts,
e.g. 512 -> 448) while leaving the MTP bank untouched (still 512 experts on
disk). vLLM derives the MTP draft MoE width from the same single
`text_config.num_experts` field (schema has no per-MTP override), so the draft
would be built 448-wide against 512-wide checkpoint tensors.
Fix: env-gated override. Serve with  VLLM_MTP_NUM_EXPERTS=512  to make the
draft build with the MTP bank's true expert count during construction; the
config value is restored immediately after, so the main model and any later
config readers are unaffected. (On NVIDIA, load_weights' n_routed_experts use
is an AMD/AITER-only pass-through and needs no override.)

Loaded via PYTHONPATH by `vllm serve` and its spawned EngineCore processes.
"""
import os
import traceback

# ---------------------------------------------------------------- Patch 1 (PLE)
try:
    import vllm.models.qwen4_exp.nvidia.ple_layer as _ple

    if hasattr(_ple, "_get_ple_embedding_quant_method"):
        _orig_get_ple_embedding_quant_method = _ple._get_ple_embedding_quant_method

        def _patched_get_ple_embedding_quant_method(*args, **kwargs):
            method = _orig_get_ple_embedding_quant_method(*args, **kwargs)
            if method is None:
                quant_config = args[0] if args else kwargs.get("quant_config")
                prefix = (
                    args[1] if len(args) > 1 else kwargs.get("prefix", "")
                )
                if (
                    prefix.endswith("ngram_embedding")
                    and type(quant_config).__name__.startswith("ModelOpt")
                ):
                    return _ple.Qwen4ExpPLEFp8EmbeddingMethod()
            return method

        _ple._get_ple_embedding_quant_method = (
            _patched_get_ple_embedding_quant_method
        )
except Exception:
    # Never break unrelated python processes that inherit this PYTHONPATH.
    traceback.print_exc()

# ------------------------------------------------- Patch 2 (MTP expert count)
_MTP_NUM_EXPERTS = int(os.environ.get("VLLM_MTP_NUM_EXPERTS", "0") or 0)

if _MTP_NUM_EXPERTS > 0:
    try:
        import functools

        from vllm.models.qwen4_exp.nvidia import mtp as _mtp_mod

        def _wrap_draft_init(cls):
            _orig_init = cls.__init__

            # functools.wraps sets __wrapped__, which inspect.signature()
            # follows: vLLM's initialize_model() introspects the __init__
            # parameter names to pick the new-style construction path
            # (model_class(vllm_config=..., prefix=...)); a bare *args/**kwargs
            # wrapper hides them and drops it into the legacy no-arg path.
            @functools.wraps(_orig_init)
            def _init(self, *args, **kwargs):
                vllm_config = kwargs.get("vllm_config") or (
                    args[0] if args else None
                )
                cfg = getattr(
                    getattr(vllm_config, "model_config", None),
                    "hf_text_config",
                    None,
                )
                saved = getattr(cfg, "num_experts", None)
                if (
                    cfg is not None
                    and saved is not None
                    and saved != _MTP_NUM_EXPERTS
                ):
                    cfg.num_experts = _MTP_NUM_EXPERTS
                    try:
                        return _orig_init(self, *args, **kwargs)
                    finally:
                        cfg.num_experts = saved
                return _orig_init(self, *args, **kwargs)

            cls.__init__ = _init

        for _cls_name in ("Qwen4ExpMultiTokenPredictor", "Qwen4ExpMTP"):
            _cls = getattr(_mtp_mod, _cls_name, None)
            if _cls is not None:
                _wrap_draft_init(_cls)

        # The draft vllm_config's model_config is a FRESH ModelConfig (built by
        # SpeculativeConfig for method=mtp) whose hf_text_config is a distinct
        # object from the target's. The draft decoder layers are constructed
        # from THIS object (Qwen4ExpDecoderLayer(draft_vllm_config, ...)), so
        # the __init__ wraps above (which cover the target-side text config)
        # are not enough. Override the draft's own text config here; it is
        # draft-only, so the override persists for the draft's lifetime.
        _orig_make_draft = _mtp_mod._make_draft_vllm_config

        @functools.wraps(_orig_make_draft)
        def _make_draft(vllm_config, mtp_start_layer_idx, *args, **kwargs):
            draft = _orig_make_draft(
                vllm_config, mtp_start_layer_idx, *args, **kwargs
            )
            try:
                draft_tc = draft.model_config.hf_text_config
                target_tc = vllm_config.model_config.hf_text_config
            except Exception:
                return draft
            if (
                draft_tc is not target_tc
                and getattr(draft_tc, "num_experts", None)
                not in (None, _MTP_NUM_EXPERTS)
            ):
                draft_tc.num_experts = _MTP_NUM_EXPERTS
            return draft

        _mtp_mod._make_draft_vllm_config = _make_draft
    except Exception:
        # Never break unrelated python processes that inherit this PYTHONPATH.
        traceback.print_exc()
