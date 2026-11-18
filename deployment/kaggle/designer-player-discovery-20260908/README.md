# Discovery-factorial designer-player candidate

Separate successor to `designer-player-reasoning-20260907`; previous evidence is unchanged. This candidate tests two independent prompt factors: a host-rendered measured-evidence block and one execution-generated, API-faithful discovery demonstration.

All four arms share the same working-belief workflow, tools, prediction/action gates, optional world-model construction, deadlines, and 16384-token ceiling:

- A: common fresh baseline
- B: measured evidence only
- C: discovery demonstration only
- D: both factors

The two-decision fixture under `fixture-artifacts/` generated the checked-in demonstration after exercising the actual helper methods and real submit-then-host-transition boundary. It is not a hand-written transcript. `last_transition()` and `changed_region()` remain measured evidence; hypotheses and objectives remain agent-authored.

Build an arm-specific notebook with:

```text
python3 build_notebook.py --discovery-arm a|b|c|d --thinking-on --max-tokens 16384
```

Local checks validate the fixture, factor separation, manifest, and generated notebook. OpenRouter screens compatibility only; compare any promising arm with the fresh A control on the target local Qwen/SGLang runtime before deployment. No Kaggle upload or launch is performed by preparation checks.
