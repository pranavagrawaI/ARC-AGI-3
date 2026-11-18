# Prompt reconciliation: CPU source and corpus evidence

The historical SGLang prompt count exceeds the old local audit by 11 tokens because SGLang serializes parsed tool models before rendering. It is not a missing default system message. The correction reproduces all 1,369 historical prompt counts exactly. No model, engine or GPU was loaded; trials 1–4 were read-only inputs.

## Decisive source path

The control wheel SHA is `d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225`, checked against trial-3's preserved package contract. Exact extracted sources and hashes are retained here.

- The preserved control argv selects the model's `.jinja` file. `source/template_manager.py:202–203,264–271` loads that file and sets `chat_template_name=None`. `source/serving_chat.py:1202–1205` consequently selects the Jinja path, not the conversation-template defaults. Every corpus request already starts with a system message.
- `source/serving_chat.py:1156` passes `item.model_dump()` for each tool. `source/protocol.py:742–769` defines the actual `Function` and `Tool` classes: the function includes `strict=False`; its serializer removes absent function-level `defer_loading`, while the outer Tool retains `defer_loading=None`.
- `source/serving_chat.py:1402–1408` passes these dictionaries to the model template. The template's tool block serializes the complete dictionary with `tojson`.

`audit.py` executes those two exact class definitions extracted from the pinned wheel, without importing SGLang. On the first real prefix (`ar25:70`), raw tools produce 108448 prompt tokens; the exact parsed-tool serialization produces 108459, matching the historical count. The ablation in `ablation.json` attributes five additional tokens to `strict: false` and six to outer `defer_loading: null`. Reordering the function keys changes rendered bytes but adds no tokens on this record.

## Corpus result

`receipt.json` binds the wheel, source, processor/tokenizer assets, frozen corpus/catalog, prior rendering audit, audit program, per-record output and proposed template by SHA-256.

- Historical minus old local count: 11 on all 1,369 records.
- Historical minus normalized local count: zero on all 1,369 records.
- Both downloaded tokenizers produce identical token ID sequences after normalization on every record; both original model templates render identical normalized bytes.
- All requests contain the same one tool schema. No initial system message is missing.
- Largest normalized prompt plus recorded completion cap: 129998, below context 139264. Image expansion remains the prior configuration-derived 400 tokens per 640×640 image.
- The proposed shared template gives the same bytes from raw corpus tools and SGLang-expanded tools on all 1,369 records.

The complete audit took 231.4 seconds on CPU. The local versions were tokenizers 0.23.3, Jinja2 3.1.6 and Pydantic 2.14.1. The historical runtime lock names Pydantic 2.14.0b2; that version difference is disclosed rather than treated as runtime equivalence.

## Candidate contrast and shared normalization

Exact upstream vLLM revision `e9757321527ca1ecd514c07c1418dd2c53da3d19` sources are retained under `source-vllm/`, with official source URLs and hashes in its manifest. `vllm/renderers/online_renderer.py:235,669` dumps tool models and passes those dictionaries to rendering. `vllm/entrypoints/generate/base/protocol.py:243–260` removes absent function `strict` and `defer_loading`; `vllm/entrypoints/openai/chat_completion/protocol.py:184–201` removes absent outer `defer_loading`.

`candidate_tools.py` executes the exact candidate base/tool/function class definitions on the corpus's unique tool schema. Its serialized result matches the original corpus tool JSON, including key order. Sending SGLang-expanded fields in the wire payload alone is insufficient: candidate parsing removes the outer null, yielding 108453 rather than 108459 on the first prefix. `candidate-tools-receipt.json` records that probe.

For a future comparison requiring identical rendered prompts, the concrete proposal is to load the same `chat_template.normalized.jinja` in BOTH arms. It changes only the tool-JSON serialization expression to an explicit field order/default representation matching the shipping control; messages, system text, tools' descriptions/parameters and images remain unchanged. It yields the shipping-control rendering from either frontend's tool dictionaries. This proposal is scoped to the verified existing corpus tool schema; it is not a general-purpose serializer for arbitrary tool extensions. It has not been applied to any prior catalog or running server.

Template SHA-256: `b4186da57f126f07c6b27e033cee286892e23c17d4dcad33d4a4d83f2d5b9206`.

The parent elected to load this normalized template explicitly for the first candidate-only feasibility packet, a visible configuration change toward the intended future matched prompt. `functional_audit.py` separately audits the six actual serialized feasibility case bodies through both exact frontend tool schemas and both tokenizers. `functional-receipt.json` binds each request byte hash to expected rendering, token counts, image hashes and grids; rendered texts are retained in `functional-renders/`. Do not silently add a system message or normalize one arm alone in a paired comparison. A runtime discrepancy remains feasibility evidence and cannot support a matched speed comparison.

## Remaining runtime evidence

This audit establishes a source-supported explanation and complete count reconciliation, not captured historical rendered bytes. Installed source/overlay identities, actual HTTP preprocessing, native rendered-prefix hashes, processed image grids/pixel values, actual native prompt counts, memory fit and functional output still require the separately authorized runtime test. The future native receipt must match the intended template and source-record expectations; unexplained discrepancies stop a speed comparison. Modified functional probes need their own rendering/count expectations, not the original replay record's count.

## Reproduce locally

Both commands write only under this audit directory and use cached CPU dependencies. Run from the repository root:

```bash
timeout 1800s uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pydantic==2.14.1 python -B analysis/serving-alternatives-20261021/trial-5-serving-screen/prompt-audit/audit.py
timeout 1800s uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pydantic==2.14.1 python -B analysis/serving-alternatives-20261021/trial-5-serving-screen/prompt-audit/candidate_tools.py
timeout 1800s uv run --offline --no-project --python 3.13 --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pydantic==2.14.1 python -B analysis/serving-alternatives-20261021/trial-5-serving-screen/prompt-audit/functional_audit.py
```

The three small supplemental candidate source files were downloaded read-only from the pinned official upstream revision; no weights were downloaded. The candidate chat protocol was copied after hash verification from the native worker's exact-source receipt. No GPU allocation, launch, staging, commit or slot request occurred in this audit.
