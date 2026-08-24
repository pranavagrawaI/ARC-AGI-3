# Gemma 4 31B V12 factored compatibility: after action

## Judgment

The run is valid and the factorization worked. Both arms passed immutable identifier binding and isolated RLE interpretation. Gemma can bind `L0:T1` amid competing identifiers, and it can decode the target RLE value as `5`.

The overall CodeAct treatment still failed the recomposed task under the frozen three-call ceiling, while Native passed it. This is real whole-interface evidence against the present CodeAct treatment for Gemma, but it is **not evidence that Gemma failed the cognitive composition after seeing the data**. CodeAct used its three calls to discover the injected-object convention, repair the query, and finally disclose the correct `L0:T1` RLE record. It never attempted decoding, grounding, or `return_result` in G3.

Therefore the machine disposition `rejected_composition` is accurate at the integrated-system level. The narrower mechanism is **CodeAct protocol overhead consumed the composition budget before cognitive composition began**.

## Causal chain

```text
G0 termination
    Native    1 call                                      pass
    CodeAct   1 call                                      pass

G1 capability retrieval
    Native    L0:T3                                       pass
    CodeAct   L0:T3                                       pass

G2A immutable identifier interference
    Native    L0:T1, 1 call, 0 Python cells              pass
    CodeAct   L0:T1, 2 calls, 1 unnecessary Python cell  pass

G2B immutable RLE interpretation
    Native    5, 1 call                                  pass
    CodeAct   5, 2 calls, 1 useful Python cell           pass

G3 recomposition
    Native    query -> 11->5 + L0:T1, 2 calls            pass

    CodeAct   import evidence                             failure
              incomplete evidence.query                  rejection
              exact query -> disclosed L0:T1 RLE record  success
              no call remained for decoding or return    fail
```

## Integrity and cost

- Nineteen calls were dispatched and accounted.
- Cost: USD 0.00267023.
- Tokens: 13,686 prompt and 4,501 completion.
- Every response used OpenInference and `google/gemma-4-31b-it`.
- No provider, route, budget, input-cap, or authority violation occurred.

## Consequence

Retire identifier confusion and RLE incompetence as explanations for this run. Do not add a decoder, raw representation, fourth call, or more discoverability and then relabel the result as the same experiment.

For the actual engineering choice, Native currently has the stronger evidence: with the same information and call ceiling it completed G3, while CodeAct spent the entire budget operating its protocol. This is precisely the cost that matters when the production system must support smaller models. A future four-call CodeAct diagnostic could show that one extra turn completes the known trajectory, but it would answer a cost-sensitivity question—not rescue V12 eligibility.
