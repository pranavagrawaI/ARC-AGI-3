# DeepSeek V4 Flash compatibility: after action

## Judgment

DeepSeek V4 Flash reproduced Kimi K3's V4 Gate 1 failure mechanism under the unchanged ladder. Both Native and CodeAct passed termination on their first calls. Native then queried once and returned the correct metadata answer in two calls. CodeAct used structured transport correctly for all three turns but its first query omitted `kind` and `grid_encoding`; T1 rejected it. DeepSeek corrected the complete query on its next turn, but the rejected attempt had already consumed the one-query allowance. Its final typed answer was correct, while zero evidence disclosures succeeded, so the frozen gate failed.

Seven accounted calls cost USD 0.000905460 through the exact `deepinfra/fp8` route. Route, provider, authority, accounting, and trace integrity passed.

## Cross-model evidence

Kimi K3 and DeepSeek V4 Flash independently produced the same causal sequence:

1. valid structured `execute_python` transport;
2. incomplete first metadata query;
3. T1 reports the exact required query;
4. model emits the exact corrected query;
5. T1 rejects it because the invalid attempt consumed the allowance;
6. model returns the typed answer correctly.

The specific omissions differed, but the failure was determined by the same gate semantics. This is now evidence against interpreting Gate 1 as model incapability. It is a compatibility-test confound that disproportionately punishes models which repair after feedback.

## Consequence

Do not advance either model under V4, because the precommitted gate failed. Do revise the next benchmark version before drawing conclusions about smaller-model viability:

- metadata projection must not require an irrelevant grid encoding;
- rejected requests and successful evidence disclosures must be counted separately;
- attempts must remain bounded;
- both arms must receive the same normalized semantic contract.

That is a benchmark repair, not a relaxation of solver authority.
